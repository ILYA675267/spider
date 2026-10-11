"""Сцена: камера 9:16, свет, тень на полу, настройки рендера.

Фон рендерится прозрачным: персонаж потом накладывается на картинку фона
(assets/background.png), которую можно заменить на любую свою.
"""
import math

import bpy


def setup(width=1080, height=1920, fps=24, samples=16, preview=False):
    s = bpy.context.scene
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.fps = fps
    s.render.film_transparent = True
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA"
    s.view_settings.view_transform = "Standard"
    s.view_settings.exposure = -0.6

    if preview:  # быстрый черновик без света и паутины — только формы
        s.render.engine = "BLENDER_WORKBENCH"
        s.display.shading.light = "STUDIO"
        s.display.shading.color_type = "MATERIAL"
        s.display.shading.show_shadows = True
        s.display.shading.show_cavity = True
        for ob in s.objects:
            if ob.get("is_outline"):
                ob.hide_render = True
    else:
        s.render.engine = "CYCLES"
        c = s.cycles
        c.device = "CPU"
        c.samples = samples
        c.use_adaptive_sampling = True
        c.adaptive_threshold = 0.05
        c.use_denoising = True
        c.denoiser = "OPENIMAGEDENOISE"
        c.max_bounces = 4
        c.diffuse_bounces = 2
        c.glossy_bounces = 1
        c.transmission_bounces = 0
        c.transparent_max_bounces = 8
        c.caustics_reflective = False
        c.caustics_refractive = False
        s.render.use_persistent_data = True

    # мир — мягкий тёплый свет заката
    w = bpy.data.worlds.new("world")
    s.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.45, 0.55, 1)
    bg.inputs["Strength"].default_value = 0.45

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    s.collection.objects.link(cam)
    cam.location = (0, -4.6, 1.02)
    cam.rotation_euler = (math.radians(90), 0, 0)
    s.camera = cam

    def light(name, kind, loc, rot, energy, color, size=2.0):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "AREA":
            ld.size = size
        ob = bpy.data.objects.new(name, ld)
        s.collection.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = [math.radians(a) for a in rot]
        return ob

    light("key", "AREA", (-2.2, -3.0, 3.2), (50, 0, -35), 700, (1.0, 0.92, 0.85), 2.5)
    light("fill", "AREA", (2.8, -2.4, 1.4), (75, 0, 50), 220, (0.75, 0.82, 1.0), 3.0)
    light("rim", "AREA", (1.6, 2.4, 2.8), (-55, 0, 145), 650, (1.0, 0.7, 0.45), 1.5)

    # невидимый пол, на котором видна только тень персонажа
    me = bpy.data.meshes.new("floor")
    me.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("floor", me)
    s.collection.objects.link(floor)
    floor.is_shadow_catcher = True
    floor.hide_render = preview
    return cam
