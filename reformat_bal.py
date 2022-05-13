import sys
from collections import defaultdict

def main():
    with open(sys.argv[1], 'r') as data:
        lines = data.readlines()

    num_images, num_points, num_observations = map(int, lines[0].split())


    map_points = {}
    problemo_count = 0
    for i in range(1, num_observations + 1):
        cam_index, point_index, img_x, img_y = map(int, lines[i].split())
        if point_index not in map_points.keys():
            map_points[point_index] = {}
        if cam_index in map_points[point_index].keys():
            problemo_count += 1
            # print('problemo at', cam_index, point_index)
        map_points[point_index][cam_index] = [img_x, img_y]


    map_points = {k: v for k, v in sorted(map_points.items(), key = lambda item : item[0])}
    # print(len(map_points))
    # print(sum([len(map_points[k]) for k, v in map_points.items()]))
    # print(problemo_count)
    # print(problemo_count + sum([len(map_points[k]) for k, v in map_points.items()]))
    actual_observation_count = sum([len(map_points[k]) for k, v in map_points.items()])
    print(num_images, num_points, actual_observation_count)

    for point_index, mapping in map_points.items():

        mapping = {k: v for k, v in sorted(mapping.items(), key = lambda item : item[0])}
        for cam_index, img_coords in mapping.items():
            print(cam_index, point_index, img_coords[0], img_coords[1])

    i = num_observations + 1
    while i < len(lines):
        print(lines[i].strip())
        i += 1

main()