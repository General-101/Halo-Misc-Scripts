import os 
import bpy
from io_scene_halo.file_jma import export_jma, import_jma
from io_scene_halo.global_functions import mesh_processing

def get_export_version(game_title):
    version = 16392
    if not game_title == 'halo1':
        version = 16395

    return version

extension_list = ('.JMA', '.JMM', '.JMT', '.JMO', '.JMR', '.JMRX', '.JMH', '.JMZ', '.JMW')
intermediate_directory = r""

game_title = 'halo1'
export_version = get_export_version(game_title)

context = bpy.context
scene = context.scene

for file_item in os.listdir(intermediate_directory):
    for extension in extension_list:
        if file_item.lower().endswith(extension.lower()):
            output_path = os.path.join(intermediate_directory, "output")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            input_filepath = os.path.join(intermediate_directory, file_item)
            output_filepath = os.path.join(output_path, file_item.rsplit('.', 1)[0])
            active_object = bpy.context.view_layer.objects.active
            
            mesh_processing.select_object(context, bpy.data.objects['source'])
            import_jma.load_file(context, input_filepath, "auto", 20011115, 0, True, False, "", "", 0, 0, print)
            mesh_processing.deselect_objects(context)

            mesh_processing.select_object(context, bpy.data.objects['output'])
            
            bpy.ops.object.mode_set(mode = 'POSE')
            scene.keemap_settings.start_frame_to_apply = scene.frame_start
            scene.keemap_settings.number_of_frames_to_apply = scene.frame_end
            bpy.ops.wm.perform_animation_transfer()
            bpy.ops.object.mode_set(mode = 'OBJECT')
            
            export_jma.write_file(context, output_filepath, print, extension, export_version, game_title, True, False, False, False, 30, 1.0 )
            mesh_processing.deselect_objects(context)
            
            break