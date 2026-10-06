import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
DEFAULT_DETECTOR = '/home/lenovo/task1_ws/py/task1_node.py'

def generate_launch_description():
    share = get_package_share_directory('vision_task1_sim')
    world = os.path.join(share, 'worlds', 'block_scene.world')
    models = os.path.join(share, 'models')
    gui = LaunchConfiguration('gui')
    detector = LaunchConfiguration('detector')
    detector_script = LaunchConfiguration('detector_script')
    soft_render = LaunchConfiguration('soft_render')
    gzserver = ExecuteProcess(cmd=['gzserver', world, '-slibgazebo_ros_init.so', '-slibgazebo_ros_factory.so', '-slibgazebo_ros_force_system.so', '-slibgazebo_ros_state.so'], output='screen', additional_env={'LIBGL_ALWAYS_SOFTWARE': soft_render})
    gzclient = ExecuteProcess(cmd=['gzclient'], output='screen', condition=IfCondition(gui), additional_env={'LIBGL_ALWAYS_SOFTWARE': soft_render})

    def spawn_block(entity, sdf_name, x, y, yaw, period):
        return TimerAction(period=period, actions=[Node(package='gazebo_ros', executable='spawn_entity.py', name='spawn_' + entity, output='screen', arguments=['-file', os.path.join(models, sdf_name), '-entity', entity, '-x', str(x), '-y', str(y), '-z', '0.025', '-Y', str(yaw)])])
    import math
    Y_RED = math.atan2(-0.09, 0.4)
    Y_BLUE = math.atan2(0.09, 0.4)
    spawn_red = spawn_block('block_red', 'block_red.sdf', 0.4, -0.09, Y_RED, 4.0)
    spawn_blue = spawn_block('block_blue', 'block_blue.sdf', 0.4, 0.09, Y_BLUE, 5.0)
    spawn_rb = spawn_block('block_red_blue', 'block_red_blue.sdf', 0.6, 0.0, 0.0, 6.0)
    det = ExecuteProcess(cmd=['python3', detector_script, '--ros-args', '-r', 'image_raw:=/camera/image_raw', '-r', 'camera_info:=/camera/camera_info'], output='screen', condition=IfCondition(detector))
    ready_banner = TimerAction(period=8.0, actions=[ExecuteProcess(cmd=['bash', '-c', 'echo; echo "========================================================"; echo "  [就绪] 地面 + 相机 + 三个方块都已在位"; echo "         现在相机画面上应该能看到 蓝 / 红蓝相间 / 红 三个方块"; echo ""; echo "  看结果（本终端被仿真占着，请新开一个任务/终端）："; echo "    任务 (9)  看 /block_info 话题频率"; echo "    任务 (10) 看 /block_info 识别结果"; echo "    任务 (11) 看相机画面（弹窗里选 /camera/image_raw）"; echo "    任务 (3)  独立真值核对（精度表格）"; echo "  停止仿真：任务 (13)  或直接关掉本终端"; echo "========================================================"; echo'], output='screen')])
    return LaunchDescription([DeclareLaunchArgument('gui', default_value='false', description='是否启动 Gazebo 3D 界面（软件渲染下会让相机更慢）'), DeclareLaunchArgument('detector', default_value='true', description='是否顺便启动识别节点'), DeclareLaunchArgument('detector_script', default_value=DEFAULT_DETECTOR, description='识别节点脚本的路径'), DeclareLaunchArgument('soft_render', default_value='1', description='1=软件渲染（WSL 上必须，否则相机一渲染就崩）'), gzserver, gzclient, spawn_red, spawn_blue, spawn_rb, det, ready_banner])
