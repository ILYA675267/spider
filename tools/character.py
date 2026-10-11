"""3D-персонаж (чиби-паук), собранный из простых форм в Blender.

Персонаж смотрит на камеру (в сторону -Y), Z — вверх, ступни на z=0.
Скелет — иерархия «шарниров» (пустых объектов): повернул шарнир — двинулась
вся часть тела ниже него. Имена шарниров — в PIVOTS, ими пользуется poses.py.
"""
import math
import os

import bmesh
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")

RED = (0.50, 0.012, 0.012, 1)
BLUE = (0.01, 0.045, 0.42, 1)
WEB = (0.012, 0.0, 0.0, 1)
BLACK = (0.006, 0.006, 0.008, 1)
WHITE = (0.92, 0.93, 0.95, 1)

OUTLINE = 0.011  # толщина чёрного контура (как у рисованного мультика)


# ---------------------------------------------------------------- материалы
def _node(nt, kind, loc, **inputs):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in inputs.items():
        setattr(n, k, v)
    return n


def _math(nt, op, a, b=None, loc=(0, 0)):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.location = loc
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _principled(nt, color_socket_or_value, rough=0.55, spec=0.35):
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if isinstance(color_socket_or_value, tuple):
        bsdf.inputs["Base Color"].default_value = color_socket_or_value
    else:
        nt.links.new(color_socket_or_value, bsdf.inputs["Base Color"])
    return bsdf


def flat_material(name, color, rough=0.55, spec=0.35, emission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = color
    bsdf = _principled(m.node_tree, color, rough, spec)
    if emission:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    return m


def web_material(name, base=RED, mode="sphere", meridians=8, rings=9.0,
                 width=0.075, rotate=(0, 0, 0), blue_sides=False, emblem=None):
    """Красная ткань с чёрной паутиной.

    mode="sphere": паутина расходится из «полюса» по оси Z (для головы — из центра лица);
    mode="cyl": продольные линии + кольца вдоль Z (руки, ноги, туловище).
    """
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = base
    nt = m.node_tree
    tc = _node(nt, "ShaderNodeTexCoord", (-1400, 0))
    mp = _node(nt, "ShaderNodeMapping", (-1200, 0))
    mp.inputs["Rotation"].default_value = [math.radians(a) for a in rotate]
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    sep = _node(nt, "ShaderNodeSeparateXYZ", (-1000, 0))
    nt.links.new(mp.outputs[0], sep.inputs[0])
    x, y, z = sep.outputs

    ang = _math(nt, "ARCTAN2", y, x, (-800, 200))
    if mode == "sphere":
        ln = _node(nt, "ShaderNodeVectorMath", (-1000, -250), operation="LENGTH")
        nt.links.new(mp.outputs[0], ln.inputs[0])
        cz = _math(nt, "DIVIDE", z, ln.outputs["Value"], (-800, -100))
        t = _math(nt, "ARCCOSINE", cz, None, (-600, -100))
        shrink = _math(nt, "SINE", t, None, (-400, 100))  # линии сходятся к полюсу
    else:
        t = z
        shrink = None
    a = _math(nt, "ABSOLUTE", _math(nt, "SINE", _math(nt, "MULTIPLY", ang, meridians, (-600, 300))), None, (-300, 300))
    if shrink is not None:
        a = _math(nt, "MULTIPLY", a, _math(nt, "ADD", shrink, 0.05), (-150, 300))
    b = _math(nt, "ABSOLUTE", _math(nt, "SINE", _math(nt, "MULTIPLY", t, rings, (-500, -200))), None, (-300, -200))
    d = _math(nt, "MINIMUM", a, b, (0, 0))
    rng = _node(nt, "ShaderNodeMapRange", (200, 0), interpolation_type="SMOOTHSTEP")
    rng.inputs["From Min"].default_value = width * 0.35
    rng.inputs["From Max"].default_value = width
    rng.inputs["To Min"].default_value = 1.0
    rng.inputs["To Max"].default_value = 0.0
    nt.links.new(d, rng.inputs["Value"])
    line = rng.outputs[0]

    mix = _node(nt, "ShaderNodeMix", (450, 0), data_type="RGBA")
    mix.inputs["A"].default_value = base
    mix.inputs["B"].default_value = WEB
    nt.links.new(line, mix.inputs["Factor"])
    color = mix.outputs["Result"]

    if blue_sides:
        # синие вставки по бокам туловища (где |x| большой относительно радиуса)
        r = _node(nt, "ShaderNodeVectorMath", (-800, -500), operation="LENGTH")
        cmb = _node(nt, "ShaderNodeCombineXYZ", (-1000, -500))
        nt.links.new(x, cmb.inputs[0])
        nt.links.new(y, cmb.inputs[1])
        nt.links.new(cmb.outputs[0], r.inputs[0])
        side = _math(nt, "DIVIDE", _math(nt, "ABSOLUTE", x), r.outputs["Value"], (-600, -500))
        sm = _node(nt, "ShaderNodeMapRange", (400, -400), interpolation_type="SMOOTHSTEP")
        sm.inputs["From Min"].default_value = 0.93
        sm.inputs["From Max"].default_value = 0.95
        nt.links.new(side, sm.inputs["Value"])
        mb = _node(nt, "ShaderNodeMix", (650, -100), data_type="RGBA")
        nt.links.new(sm.outputs[0], mb.inputs["Factor"])
        nt.links.new(color, mb.inputs["A"])
        mb.inputs["B"].default_value = BLUE
        color = mb.outputs["Result"]

    if emblem:
        # паук на груди: картинка, спроецированная спереди (только на передней стороне)
        img = bpy.data.images.load(emblem["path"], check_existing=True)
        cmb = _node(nt, "ShaderNodeCombineXYZ", (-600, -800))
        u = _math(nt, "ADD", _math(nt, "DIVIDE", x, emblem["size"]), 0.5, (-800, -750))
        v = _math(nt, "ADD", _math(nt, "DIVIDE", _math(nt, "SUBTRACT", z, emblem["z"]), emblem["size"]), 0.5, (-800, -900))
        nt.links.new(u, cmb.inputs[0])
        nt.links.new(v, cmb.inputs[1])
        tex = _node(nt, "ShaderNodeTexImage", (-400, -800), extension="CLIP", interpolation="Cubic")
        tex.image = img
        nt.links.new(cmb.outputs[0], tex.inputs[0])
        front = _math(nt, "LESS_THAN", y, 0.0, (-400, -1000))
        fac = _math(nt, "MULTIPLY", tex.outputs["Alpha"], front, (-150, -900))
        me = _node(nt, "ShaderNodeMix", (850, -200), data_type="RGBA")
        nt.links.new(fac, me.inputs["Factor"])
        nt.links.new(color, me.inputs["A"])
        me.inputs["B"].default_value = WEB
        color = me.outputs["Result"]

    _principled(nt, color)
    return m


def outline_material():
    """Чёрный контур методом «вывернутой оболочки»: видна только задняя сторона."""
    m = bpy.data.materials.get("outline")
    if m:
        return m
    m = bpy.data.materials.new("outline")
    m.use_nodes = True
    m.diffuse_color = BLACK
    nt = m.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", (400, 0))
    geo = _node(nt, "ShaderNodeNewGeometry", (-300, 200))
    em = _node(nt, "ShaderNodeEmission", (-100, -100))
    em.inputs["Color"].default_value = BLACK
    tr = _node(nt, "ShaderNodeBsdfTransparent", (-100, 100))
    mix = _node(nt, "ShaderNodeMixShader", (150, 0))
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------- геометрия
def _mesh_obj(name, bm, parent, loc=(0, 0, 0), mat=None, outline=True, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = loc
    if mat:
        me.materials.append(mat)
    if outline:
        hull = bpy.data.objects.new(name + "_line", me)
        bpy.context.scene.collection.objects.link(hull)
        hull.parent = ob
        mod = hull.modifiers.new("grow", "DISPLACE")
        mod.mid_level = 0.0
        mod.strength = OUTLINE
        hull.material_slots[0].link = "OBJECT"
        hull.material_slots[0].material = outline_material()
        hull.visible_shadow = False
        hull.visible_diffuse = False
        hull.visible_glossy = False
        hull.visible_transmission = False
        hull.visible_volume_scatter = False
        hull["is_outline"] = True
    return ob


def capsule(radius, length, scale=(1, 1, 1), segs=32, rings=16):
    """Капсула вдоль Z с центром в нуле: length — расстояние между центрами полусфер."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=radius)
    for v in bm.verts:
        if v.co.z > 1e-5:
            v.co.z += length / 2
        elif v.co.z < -1e-5:
            v.co.z -= length / 2
        v.co.x *= scale[0]
        v.co.y *= scale[1]
        v.co.z *= scale[2]
    return bm


def ellipsoid(r, scale=(1, 1, 1), segs=48, rings=24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=r)
    for v in bm.verts:
        v.co = Vector((v.co.x * scale[0], v.co.y * scale[1], v.co.z * scale[2]))
    return bm


def pivot(name, parent, loc, mode="XYZ"):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(e)
    e.parent = parent
    e.location = loc
    e.rotation_mode = mode
    return e


def eye_outline(n=72):
    """Контур глаза (левого): капля с острым внешним верхним уголком. u — наружу, v — вверх."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = 1.0 + 0.55 * max(0.0, math.cos(t - math.radians(42))) ** 8
        u, v = 0.55 * r * math.cos(t), 0.62 * r * math.sin(t)
        # наклон: внешний край выше внутреннего
        a = math.radians(16)
        pts.append((u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)))
    return pts


def eye_patch(side, hr, sc, k, lift, cx=0.155, cz=0.04, size=0.235, rings=10):
    """Пятно глаза на поверхности эллипсоида головы (центр головы в 0)."""
    A, B, C = hr * sc[0], hr * sc[1], hr * sc[2]
    outline = eye_outline()
    bm = bmesh.new()

    def surf(u, v):
        x = side * (cx + u * size * k)
        z = cz + v * size * k
        q = 1 - (x / A) ** 2 - (z / C) ** 2
        q = max(q, 0.02)
        n = Vector((x / A ** 2, -B * math.sqrt(q) / B ** 2, z / C ** 2)).normalized()
        return Vector((x, -B * math.sqrt(q), z)) + n * lift

    prev = None
    for ri in range(1, rings + 1):
        f = ri / rings
        ring = [bm.verts.new(surf(u * f, v * f)) for u, v in outline]
        n = len(ring)
        for i in range(n):
            j = (i + 1) % n
            if prev is not None:
                q = (prev[i], prev[j], ring[j], ring[i])
                bm.faces.new(q[::-1] if side > 0 else q)
        if prev is None:
            bm.faces.new(ring[::-1] if side > 0 else ring)
        prev = ring
    bm.normal_update()
    return bm


# ---------------------------------------------------------------- эмблема
def make_emblem(path):
    """Рисует паука на груди (PNG с прозрачностью)."""
    from PIL import Image, ImageDraw

    S = 512
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S // 2
    w = 16
    for side in (-1, 1):
        # 4 ноги с каждой стороны: (колено, ступня)
        legs = [((60, -95), (75, -205)), ((95, -30), (175, -120)),
                ((95, 40), (170, 140)), ((60, 95), (85, 215))]
        for (kx, ky), (fx, fy) in legs:
            pts = [(c + side * 22, c + ky * 0.25), (c + side * kx, c + ky), (c + side * fx, c + fy)]
            d.line(pts, fill=(0, 0, 0, 255), width=w, joint="curve")
    d.ellipse([c - 34, c - 95, c + 34, c - 20], fill=(0, 0, 0, 255))     # голова-грудь
    d.ellipse([c - 46, c - 30, c + 46, c + 120], fill=(0, 0, 0, 255))    # брюшко
    im.save(path)
    return path


# ---------------------------------------------------------------- персонаж
def build(root_name="spidey"):
    """Строит персонажа. Возвращает dict {имя шарнира: объект}."""
    os.makedirs(ASSETS, exist_ok=True)
    emblem = make_emblem(os.path.join(ASSETS, "emblem.png"))

    m_head = web_material("m_head", mode="sphere", meridians=8, rings=7.0, width=0.07, rotate=(90, 0, 0))
    m_torso = web_material("m_torso", mode="cyl", meridians=7, rings=22.0, width=0.08, blue_sides=True,
                           emblem={"path": emblem, "size": 0.36, "z": 0.08})
    m_limb = web_material("m_limb", mode="cyl", meridians=4, rings=28.0, width=0.09)
    m_hand = web_material("m_hand", mode="sphere", meridians=5, rings=6.0, width=0.09)
    m_blue = flat_material("m_blue", BLUE, rough=0.5)
    m_black = flat_material("m_black", BLACK, rough=0.7, spec=0.2)
    m_white = flat_material("m_white", WHITE, rough=0.25, spec=0.6, emission=0.15)
    m_inner = flat_material("m_inner", (0.12, 0.0, 0.0, 1), rough=0.8)

    P = {}
    root = pivot(root_name, None, (0, 0, 0))
    P["root"] = root
    hips = P["hips"] = pivot("hips", root, (0, 0, 0.74))
    _mesh_obj("pelvis", capsule(0.17, 0.06, scale=(1.05, 0.82, 1)), hips, (0, 0, 0.04), m_blue)

    spine = P["spine"] = pivot("spine", hips, (0, 0, 0.06))
    _mesh_obj("torso", capsule(0.19, 0.24, scale=(1.08, 0.80, 1)), spine, (0, 0, 0.17), m_torso)

    neck = P["neck"] = pivot("neck", spine, (0, 0, 0.42))
    _mesh_obj("neckmesh", capsule(0.075, 0.06), neck, (0, 0, 0.03), m_limb, outline=False)

    head = P["head"] = pivot("head", neck, (0, 0, 0.08))
    hr = 0.40
    sc = (0.97, 0.92, 1.06)
    hc = hr * 0.98  # центр головы над шеей

    def in_jaw(f):  # нижняя передняя часть головы — «челюсть» маски
        c = f.calc_center_median()
        return c.z < -0.13 * hr and c.y < 0.06

    skull = ellipsoid(hr, scale=sc, segs=64, rings=32)
    bmesh.ops.delete(skull, geom=[f for f in skull.faces if in_jaw(f)], context="FACES_ONLY")
    _mesh_obj("skull", skull, head, (0, 0, hc), m_head)
    _mesh_obj("mouth_inner", ellipsoid(hr * 0.93, scale=(0.93, 0.88, 0.98)), head, (0, 0, hc),
              m_inner, outline=False)
    # нижняя «челюсть» маски на шарнире у ушей: поворот +X открывает рот
    hinge = Vector((0, 0.06, -0.10))
    jaw = P["jaw"] = pivot("jaw", head, (0, hinge.y, hc + hinge.z))
    jbm = ellipsoid(hr * 1.004, scale=sc, segs=64, rings=32)
    bmesh.ops.delete(jbm, geom=[f for f in jbm.faces if not in_jaw(f)], context="FACES_ONLY")
    bmesh.ops.delete(jbm, geom=[v for v in jbm.verts if not v.link_faces], context="VERTS")
    _mesh_obj("jawmesh", jbm, jaw, (0, -hinge.y, -hinge.z), m_head)

    # глаза: белые линзы в толстой чёрной оправе, «натянутые» на голову
    for side, nm in ((1, "eye_L"), (-1, "eye_R")):
        # шарнир — в центре глаза, чтобы моргание/прищур сжимали глаз к его середине
        ec = Vector((side * 0.155, -hr * sc[1] * 0.86, hc + 0.04))
        e = P[nm] = pivot(nm, head, ec)
        off = Vector((0, 0, hc)) - ec
        _mesh_obj(nm + "_rim", eye_patch(side, hr, sc, 1.0, 0.008), e, off, m_black, outline=False)
        _mesh_obj(nm + "_lens", eye_patch(side, hr, sc, 0.76, 0.014), e, off, m_white, outline=False, smooth=False)

    # руки
    for side, s in (("L", 1), ("R", -1)):
        sh = P["shoulder_" + side] = pivot("shoulder_" + side, spine, (s * 0.20, 0, 0.33), mode="ZYX")
        _mesh_obj("deltoid_" + side, ellipsoid(0.085), sh, (0, 0, -0.01), m_limb)
        _mesh_obj("upperarm_" + side, capsule(0.068, 0.17), sh, (0, 0, -0.105), m_limb)
        el = P["elbow_" + side] = pivot("elbow_" + side, sh, (0, 0, -0.21), mode="XYZ")
        _mesh_obj("forearm_" + side, capsule(0.07, 0.16), el, (0, 0, -0.095), m_limb)
        wr = P["wrist_" + side] = pivot("wrist_" + side, el, (0, 0, -0.20), mode="XYZ")
        _mesh_obj("palm_" + side, ellipsoid(0.085, scale=(0.7, 1.0, 1.0)), wr, (0, 0, -0.06), m_hand)
        # большой палец
        th = P["thumb_" + side] = pivot("thumb_" + side, wr, (-s * 0.0, -0.05, -0.04), mode="XYZ")
        th.rotation_euler = (math.radians(25), 0, 0)
        _mesh_obj("thumbm_" + side, capsule(0.024, 0.05), th, (0, 0, -0.035), m_hand)
        # указательный — отдельно, чтобы показывать
        ix = P["index_" + side] = pivot("index_" + side, wr, (0, -0.032, -0.115), mode="XYZ")
        _mesh_obj("indexm_" + side, capsule(0.022, 0.06), ix, (0, 0, -0.04), m_hand)
        # остальные три пальца
        for i, fy in enumerate((0.0, 0.030, 0.058)):
            f = P[f"finger{i}_{side}"] = pivot(f"finger{i}_{side}", wr, (0, fy, -0.112 + i * 0.004), mode="XYZ")
            _mesh_obj(f"fingerm{i}_{side}", capsule(0.021, 0.05 - i * 0.006), f, (0, 0, -0.035), m_hand)

    # ноги
    for side, s in (("L", 1), ("R", -1)):
        hp = P["hip_" + side] = pivot("hip_" + side, hips, (s * 0.095, 0, 0.0), mode="XYZ")
        _mesh_obj("thigh_" + side, capsule(0.092, 0.17), hp, (0, 0, -0.12), m_blue)
        kn = P["knee_" + side] = pivot("knee_" + side, hp, (0, 0, -0.27), mode="XYZ")
        _mesh_obj("shin_" + side, capsule(0.085, 0.22), kn, (0, 0, -0.16), m_limb)
        an = P["ankle_" + side] = pivot("ankle_" + side, kn, (0, 0, -0.36), mode="XYZ")
        _mesh_obj("foot_" + side, capsule(0.085, 0.13, scale=(1.05, 0.72, 1.0)), an, (0, -0.05, -0.035), m_limb)
        an.children[0].rotation_euler = (math.radians(90), 0, 0)

    return P


if __name__ == "__main__":
    build()
