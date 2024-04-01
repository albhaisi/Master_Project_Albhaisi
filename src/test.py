from nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/dataroot', verbose=True)

my_scene = nusc.scene[0]
print(my_scene)