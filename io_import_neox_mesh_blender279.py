# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Format parser adapted from zhouhang95/neox_tools (GPL-3.0):
# https://github.com/zhouhang95/neox_tools
bl_info = {
    "name": "Import NetEase NeoX .mesh (HPMA experimental)",
    "author": "Detexki99, based on the public neox_tools mesh parser",
    "version": (0, 1, 0),
    "blender": (2, 79, 0),
    "location": "File > Import > NetEase NeoX Mesh (.mesh)",
    "description": "Imports compatible NetEase NeoX .mesh files, including UVs and basic skeletal weights",
    "category": "Import-Export",
}

import bpy
import os
import struct
import math
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector


# This reader follows the published neox_tools/converter.py layout.
# HPMA may use a variant; an unknown signature is warned about, not rejected.

class MeshFormatError(Exception):
    pass


def _read_exact(f, size, label):
    data = f.read(size)
    if len(data) != size:
        raise MeshFormatError("Unexpected end of file while reading %s" % label)
    return data


def _u8(f):
    return struct.unpack('<B', _read_exact(f, 1, 'uint8'))[0]


def _u16(f):
    return struct.unpack('<H', _read_exact(f, 2, 'uint16'))[0]


def _u32(f):
    return struct.unpack('<I', _read_exact(f, 4, 'uint32'))[0]


def _f32(f):
    return struct.unpack('<f', _read_exact(f, 4, 'float32'))[0]


def parse_neox_mesh(filepath):
    model = {'mesh_parts': []}
    file_size = os.path.getsize(filepath)
    # Safety caps prevent malformed files from requesting huge allocations.
    max_vertices = 5000000
    max_faces = 5000000
    max_bones = 100000
    max_parts = 10000

    with open(filepath, 'rb') as f:
        magic = _read_exact(f, 8, 'header')
        model['magic'] = magic
        model['bone_exist'] = _u32(f)

        bone_parents = []
        bone_names = []
        bone_matrices = []
        if model['bone_exist']:
            if model['bone_exist'] > 1:
                count = _u8(f)
                _read_exact(f, 2, 'bone LOD header')
                _read_exact(f, count * 4, 'bone LOD data')

            bone_count = _u16(f)
            if bone_count > max_bones:
                raise MeshFormatError('Implausible bone count: %d' % bone_count)
            for unused in range(bone_count):
                p = _u8(f)
                bone_parents.append(-1 if p == 255 else p)
            for unused in range(bone_count):
                raw_name = _read_exact(f, 32, 'bone name')
                name = raw_name.decode('utf-8', 'replace').replace('\x00', '').strip().replace(' ', '_')
                bone_names.append(name or ('bone_%03d' % len(bone_names)))

            bone_extra_info = _u8(f)
            if bone_extra_info:
                _read_exact(f, bone_count * 28, 'bone extra info')

            # Matrices are stored as 16 float32 values each. Translation is
            # interpreted the same way as neox_tools' IQE converter.
            for unused in range(bone_count):
                bone_matrices.append([_f32(f) for i in range(16)])

            # The converter adds a dummy root if more than one root exists.
            if bone_parents.count(-1) > 1:
                old_count = len(bone_parents)
                bone_parents = [old_count if p == -1 else p for p in bone_parents]
                bone_parents.append(-1)
                bone_names.append('dummy_root')
                bone_matrices.append([1.0, 0.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0, 0.0,
                                      0.0, 0.0, 1.0, 0.0,
                                      0.0, 0.0, 0.0, 1.0])

            flag = _u8(f)
            if flag != 0:
                raise MeshFormatError('Unexpected bone-section flag: %d' % flag)

        _offset = _u32(f)
        # Each part starts with vertex count, face count, UV-layer count,
        # and vertex-colour-layer count. uint16 value 1 marks the end.
        for part_num in range(max_parts):
            if f.tell() + 2 > file_size:
                raise MeshFormatError('Could not find end marker for mesh-part table')
            flag_pos = f.tell()
            marker = _u16(f)
            if marker == 1:
                break
            f.seek(flag_pos)
            vc = _u32(f)
            fc = _u32(f)
            uv_layers = _u8(f)
            color_len = _u8(f)
            if vc > max_vertices or fc > max_faces or uv_layers > 16 or color_len > 16:
                raise MeshFormatError('Invalid mesh-part header near byte 0x%X' % flag_pos)
            model['mesh_parts'].append((vc, fc, uv_layers, color_len))
        else:
            raise MeshFormatError('Too many mesh parts; format may differ from this parser')

        if not model['mesh_parts']:
            raise MeshFormatError('No mesh parts were found; this may be a different .mesh variant')

        vertex_count = _u32(f)
        face_count = _u32(f)
        if vertex_count == 0 or vertex_count > max_vertices:
            raise MeshFormatError('Implausible vertex count: %d' % vertex_count)
        if face_count > max_faces:
            raise MeshFormatError('Implausible face count: %d' % face_count)
        model['position'] = [(_f32(f), _f32(f), _f32(f)) for i in range(vertex_count)]
        model['normal'] = [(_f32(f), _f32(f), _f32(f)) for i in range(vertex_count)]

        normal_extra_flag = _u16(f)
        if normal_extra_flag:
            _read_exact(f, vertex_count * 12, 'extra vertex data')

        faces = []
        for i in range(face_count):
            tri = (_u16(f), _u16(f), _u16(f))
            if max(tri) >= vertex_count:
                raise MeshFormatError('Face %d references a vertex outside the mesh' % i)
            faces.append(tri)
        model['face'] = faces

        uv = []
        for part_vc, _part_fc, uv_layers, _color_len in model['mesh_parts']:
            if uv_layers > 0:
                for i in range(part_vc):
                    uv.append((_f32(f), _f32(f)))
                if uv_layers > 1:
                    _read_exact(f, part_vc * 8 * (uv_layers - 1), 'additional UV layers')
            else:
                uv.extend([(0.0, 0.0)] * part_vc)
        # Avoid indexing errors with variants whose part counts do not quite match.
        if len(uv) < vertex_count:
            uv.extend([(0.0, 0.0)] * (vertex_count - len(uv)))
        model['uv'] = uv[:vertex_count]

        for part_vc, _part_fc, _uv_layers, color_len in model['mesh_parts']:
            if color_len:
                _read_exact(f, part_vc * 4 * color_len, 'vertex colours')

        if model['bone_exist']:
            model['vertex_joint'] = []
            for i in range(vertex_count):
                model['vertex_joint'].append([_u8(f) for unused in range(4)])
            model['vertex_joint_weight'] = []
            for i in range(vertex_count):
                model['vertex_joint_weight'].append([_f32(f) for unused in range(4)])

        model['bone_parent'] = bone_parents
        model['bone_name'] = bone_names
        model['bone_matrix'] = bone_matrices

        part_face_sum = sum(part[1] for part in model['mesh_parts'])
        if part_face_sum != len(model['face']):
            # Still allow geometry import; material grouping will be approximate.
            model['part_face_counts'] = []
            remaining = len(model['face'])
            for part in model['mesh_parts']:
                n = min(part[1], remaining)
                model['part_face_counts'].append(n)
                remaining -= n
            if remaining > 0 and model['part_face_counts']:
                model['part_face_counts'][-1] += remaining
        else:
            model['part_face_counts'] = [p[1] for p in model['mesh_parts']]

    return model


def _unique_name(name, used):
    base = name or 'bone'
    candidate = base
    n = 1
    while candidate in used:
        candidate = '%s_%03d' % (base, n)
        n += 1
    used.add(candidate)
    return candidate


def _create_armature(context, model, object_name):
    if not model.get('bone_exist') or not model.get('bone_name'):
        return None, []

    names = []
    used = set()
    for raw in model['bone_name']:
        names.append(_unique_name(raw, used))

    # translation_from_matrix(matrix.T) in neox_tools corresponds to the
    # original flattened matrix's final row (items 12:15).
    heads = []
    for values in model['bone_matrix']:
        x, y, z = values[12], values[13], values[14]
        heads.append(Vector((-x, y, z)))

    children = [[] for unused in names]
    parents = list(model['bone_parent'])
    for idx, parent in enumerate(parents):
        if parent >= 0 and parent < len(names) and parent != idx:
            children[parent].append(idx)

    arm_data = bpy.data.armatures.new(object_name + '_ArmatureData')
    arm_obj = bpy.data.objects.new(object_name + '_Armature', arm_data)
    context.scene.objects.link(arm_obj)
    arm_obj.show_x_ray = True
    arm_data.show_names = False

    old_active = context.scene.objects.active
    for obj in context.selected_objects:
        obj.select = False
    arm_obj.select = True
    context.scene.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = []
    for idx, name in enumerate(names):
        eb = arm_data.edit_bones.new(name)
        eb.head = heads[idx]
        # Aim the bone toward its first child if possible; otherwise use its
        # parent's direction reversed, or a small vertical fallback.
        target = None
        valid_children = [c for c in children[idx] if (heads[c] - heads[idx]).length > 1e-5]
        if valid_children:
            target = heads[valid_children[0]]
        elif idx < len(parents) and parents[idx] >= 0 and parents[idx] < len(heads):
            parent_head = heads[parents[idx]]
            if (heads[idx] - parent_head).length > 1e-5:
                target = heads[idx] + (heads[idx] - parent_head).normalized() * 0.08
        if target is None or (target - heads[idx]).length < 1e-5:
            target = heads[idx] + Vector((0.0, 0.08, 0.0))
        eb.tail = target
        eb.use_connect = False
        edit_bones.append(eb)

    for idx, parent in enumerate(parents):
        if parent >= 0 and parent < len(edit_bones) and parent != idx:
            edit_bones[idx].parent = edit_bones[parent]

    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.select = False
    if old_active and old_active.name in context.scene.objects:
        context.scene.objects.active = old_active
    return arm_obj, names


def build_scene(context, model, filepath):
    object_name = os.path.splitext(os.path.basename(filepath))[0]
    vertices = [(-x, y, z) for x, y, z in model['position']]
    # neox_tools' OBJ/IQE conversion mirrors X and swaps the first two indices
    # to preserve triangle winding.
    faces = [(b, a, c) for (a, b, c) in model['face']]

    mesh_data = bpy.data.meshes.new(object_name + '_Mesh')
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update(calc_edges=True)
    mesh_obj = bpy.data.objects.new(object_name, mesh_data)
    context.scene.objects.link(mesh_obj)

    # UVs are written per loop, using the first channel in the NeoX mesh.
    # Blender 2.79 creates the layer through uv_textures; later versions use
    # uv_layers.new(). Creating the legacy texture also creates the UV loop map.
    if model.get('uv') and len(model['uv']) >= len(vertices):
        try:
            uv_layer = mesh_data.uv_layers.new(name='UVMap')
        except AttributeError:
            # Blender 2.79 compatibility: uv_layers is readable, but its
            # collection does not expose .new(); use the legacy API to create it.
            mesh_data.uv_textures.new(name='UVMap')
            uv_layer = mesh_data.uv_layers.active
        if uv_layer is None:
            raise RuntimeError('Could not create a UV layer with this Blender version')
        for poly in mesh_data.polygons:
            for loop_index in poly.loop_indices:
                vertex_index = mesh_data.loops[loop_index].vertex_index
                if vertex_index < len(model['uv']):
                    u, v = model['uv'][vertex_index]
                    uv_layer.data[loop_index].uv = (u, 1.0 - v)

    # Preserve NeoX submesh boundaries as material slots (texture hookup is separate).
    face_start = 0
    for part_index, face_count in enumerate(model.get('part_face_counts', [])):
        mat = bpy.data.materials.new('%s_Mat_%02d' % (object_name, part_index))
        mesh_obj.data.materials.append(mat)
        face_end = min(face_start + face_count, len(mesh_data.polygons))
        for poly_index in range(face_start, face_end):
            mesh_data.polygons[poly_index].material_index = part_index
        face_start = face_end
    if not mesh_obj.data.materials:
        mesh_obj.data.materials.append(bpy.data.materials.new(object_name + '_Mat'))

    arm_obj, bone_names = _create_armature(context, model, object_name)
    if arm_obj:
        # Vertex groups reproduce the stored skin weights where indices are valid.
        groups = [mesh_obj.vertex_groups.new(name=name) for name in bone_names]
        joints = model.get('vertex_joint', [])
        weights = model.get('vertex_joint_weight', [])
        for vertex_index in range(min(len(vertices), len(joints), len(weights))):
            for bone_index, weight in zip(joints[vertex_index], weights[vertex_index]):
                if bone_index < len(groups) and weight > 0.000001:
                    groups[bone_index].add([vertex_index], float(weight), 'ADD')

        modifier = mesh_obj.modifiers.new('NeoX Armature', 'ARMATURE')
        modifier.object = arm_obj
        mesh_obj.parent = arm_obj
        mesh_obj.matrix_parent_inverse = arm_obj.matrix_world.inverted()

    # Select the imported mesh so the result is immediately visible/usable.
    for obj in context.selected_objects:
        obj.select = False
    mesh_obj.select = True
    context.scene.objects.active = mesh_obj
    return mesh_obj, arm_obj


class IMPORT_SCENE_OT_neox_mesh_279(bpy.types.Operator, ImportHelper):
    """Import a compatible NetEase NeoX .mesh file"""
    bl_idname = 'import_scene.neox_mesh_279'
    bl_label = 'Import NeoX Mesh'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.mesh'
    filter_glob = StringProperty(default='*.mesh', options={'HIDDEN'})

    def execute(self, context):
        try:
            model = parse_neox_mesh(self.filepath)
            build_scene(context, model, self.filepath)
        except Exception as exc:
            self.report({'ERROR'}, 'NeoX .mesh import failed: %s' % exc)
            return {'CANCELLED'}

        signature = model.get('magic', b'')[:4]
        if signature != b'\x34\x80\xc8\xbb':
            self.report({'WARNING'}, 'Imported using the generic parser, but header differs from known NeoX signature; inspect geometry/rig')
        if model.get('bone_exist') and not model.get('vertex_joint'):
            self.report({'WARNING'}, 'Skeleton was read but skin weights were not found')
        self.report({'INFO'}, 'Imported %d vertices, %d triangles, %d bones' % (
            len(model['position']), len(model['face']), len(model.get('bone_name', []))))
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_neox_mesh_279.bl_idname,
                         text='NetEase NeoX Mesh (.mesh)')


def register():
    bpy.utils.register_class(IMPORT_SCENE_OT_neox_mesh_279)
    bpy.types.INFO_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.INFO_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_neox_mesh_279)


if __name__ == '__main__':
    register()
