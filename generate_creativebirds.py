import numpy as np
import matplotlib.pyplot as plt
import ndjson
import json
from PIL import Image

from utils.process_ndjson_quickdraw import Drawing


def glue(nparr1, nparr2):
    if len(nparr1) == 0:
        return nparr2
    if len(nparr2) == 0:
        return nparr1
    return np.vstack([nparr1, nparr2])


def filter_points_on_white(image, points):
    newpoints = list()
    for point in points:
        averagegray = np.mean(image[int(np.floor(point[1])):int(np.ceil(point[1]) + 1),
                              int(np.floor(point[0])):int(np.ceil(point[0] + 1))])
        # print(point, " : ", averagegray)
        if averagegray > 0.1:
            newpoints.append(point)
    return newpoints


class DatasetItem:
    def __init__(self, image, keypoints_dict, thin_image):
        self.image = image
        self.endpoints = keypoints_dict['endpoints']
        self.selfintersectionpoints = keypoints_dict['selfintersectionpoints']
        self.intersectionpoints = keypoints_dict['intersectionpoints']
        self.sharppoints = keypoints_dict['sharppoints']
        self.thinimage = thin_image

    @classmethod
    def from_drawing(cls, draw: Drawing):
        image, keypoints_dict = draw.render_image(288, line_diameter=4,
                                                  return_keypoints=True)
        new_endpoints, new_sharppoints, new_intersections = draw.get_filtered_endpoints()
        add_endpoints, upd_sharppoints = draw.get_filtered_sharp()
        keypoints_dict['endpoints'] = np.array(new_endpoints)
        keypoints_dict['sharppoints'] = np.array(upd_sharppoints)
        keypoints_dict['endpoints'] = glue(nparr1=np.array(add_endpoints),
                                           nparr2=keypoints_dict['endpoints'])
        keypoints_dict['intersectionpoints'] = glue(nparr1=np.array(new_intersections),
                                                    nparr2=keypoints_dict['intersectionpoints'])
        keypoints_dict['sharppoints'] = glue(nparr1=np.array(new_sharppoints),
                                             nparr2=np.array(upd_sharppoints))
        thin_image = draw.render_image(288, line_diameter=2,
                                       return_keypoints=False)
        return cls(image=image, keypoints_dict=keypoints_dict, thin_image=thin_image)

    @classmethod
    def from_ndjson_item(cls, item):
        draw = Drawing.from_drawing_data(item['drawing'], raw_ndjson=True, apply_rdp=False)
        return cls.from_drawing(draw=draw)

    @classmethod
    def from_npz(cls, path_npz):
        with open(path_npz, "rb") as f:
            loaded = np.load(f)
            keypoints_dict = dict()
            for key in loaded.keys():
                if key.endswith('points'):
                    keypoints_dict[key] = loaded[key]
            return cls(image=loaded['image'], thin_image=loaded['thin_image'], keypoints_dict=keypoints_dict)

    def _mindistance(self, point, np_points):
        if len(np_points)==0:
            return 100
        dists = np_points - point
        return np.min(np.linalg.norm(dists, axis=1))

    def filter_points(self):
        newendpoints = list()
        distance_thr = 3
        for point in self.endpoints:
            if (self._mindistance(point, self.intersectionpoints)>distance_thr) \
                    and (self._mindistance(point, self.selfintersectionpoints)>distance_thr):
                    # and (self._mindistance(point, self.sharppoints)>distance_thr):
                newendpoints.append(point)
        self.endpoints = np.array(newendpoints)
        newsharppoints = list()
        for point in self.sharppoints:
            if (self._mindistance(point, self.intersectionpoints)>distance_thr) \
                    and (self._mindistance(point, self.selfintersectionpoints)>distance_thr) \
                    and (self._mindistance(point, self.endpoints)>distance_thr):
                newsharppoints.append(point)
        self.sharppoints = np.array(newsharppoints)
        self.intersectionpoints = glue(nparr1=self.intersectionpoints,
                                       nparr2=self.selfintersectionpoints)
        self.selfintersectionpoints = np.array([])
        newintersections = list()
        for point in self.intersectionpoints:
            if len(newintersections)==0:
                newintersections.append(point)
                continue
            if self._mindistance(point, np_points=np.array(newintersections))<distance_thr:
                continue
            newintersections.append(point)
        self.intersectionpoints = np.array(newintersections)

    def filter_points_too_close(self, thr=4):
        """
        This dunctions removes sharppoints and intersections if they are close to a point from the same family
        """
        newsharp = list()
        for point in self.sharppoints:
            if len(newsharp)==0:
                newsharp.append(point)
                continue
            if self._mindistance(point, np_points=np.array(newsharp))>thr:
                newsharp.append(point)
        self.sharppoints = np.array(newsharp)
        newinter = list()
        for point in self.intersectionpoints:
            if len(newinter)==0:
                newinter.append(point)
                continue
            if self._mindistance(point, np_points=np.array(newinter))>thr:
                newinter.append(point)
        self.intersectionpoints = np.array(newinter)

    def update_image(self, pngpath):
        self.image = 1 - np.array(Image.open(pngpath).convert('L')).astype(np.float32) / 256.0

    def update_labels_from_json(self, path_json):
        """
        This function reads labelme jsonfile and updates object properties corresponding to json.
        1 - endpoints, 3 - intersections, 5 - sharp; selfintersections are cleared
        """
        with open(path_json) as f:
            load_dict = json.load(f)

        # print(load_dict)

        self.selfintersectionpoints = np.array(list())  # this will remain empty
        self.intersectionpoints = list()
        self.endpoints = list()
        self.sharppoints = list()

        for shape in load_dict['shapes']:
            if shape['label']=='1':
                self.endpoints.append(shape['points'][0])
            if shape['label']=='3':
                self.intersectionpoints.append(shape['points'][0])
            if shape['label']=='5':
                self.sharppoints.append(shape['points'][0])

        self.intersectionpoints = np.array(self.intersectionpoints)
        self.endpoints = np.array(self.endpoints)
        self.sharppoints = np.array(self.sharppoints)

    def plt_plot(self):
        plt.imshow(self.image, cmap='gray_r')
        if len(self.endpoints)>0:
            plt.scatter(self.endpoints[:, 0], self.endpoints[:, 1], label='endpoints', alpha=0.7, marker='x', linewidths=1)
        if len(self.selfintersectionpoints) > 0:
            plt.scatter(self.selfintersectionpoints[:, 0], self.selfintersectionpoints[:, 1], label='selfintersectionpoints', alpha=0.7, marker='x', linewidths=1)
        if len(self.intersectionpoints) > 0:
            plt.scatter(self.intersectionpoints[:, 0], self.intersectionpoints[:, 1], label='intersectionpoints', alpha=0.7, marker='x', linewidths=1)
        if len(self.sharppoints) > 0:
            plt.scatter(self.sharppoints[:, 0], self.sharppoints[:, 1], label='sharppoints', alpha=0.7, marker='x', linewidths=1)

    def dump_json(self, filename="creativebirds_labelme/test.json", imagePath='test.png'):
        points = list()

        def gen_point(point, cl):
            d = dict()
            d['label'] = str(cl)
            d['points'] = [list(point)]
            d['group_id'] = None
            d['shape_type'] = 'point'
            d['flags'] = dict()
            return d

        for point in self.endpoints:
            points.append(gen_point(point, 1))
        for point in self.sharppoints:
            points.append(gen_point(point, 5))
        for point in self.intersectionpoints:
            points.append(gen_point(point, 3))
        for point in self.selfintersectionpoints:
            points.append(gen_point(point, 3))

        with open(filename, "w") as f:
            json.dump({"version": "4.5.6", "flags": {}, "shapes": points, "imagePath": imagePath, "imageData": None}, f)

    def dump_png(self, filename="creativebirds_labelme/test.png"):
        plt.imsave(fname=filename, arr=self.image, cmap='gray_r')
        plt.close()

    def dump_npz(self, filepath):
        sample = dict()
        sample['image'] = self.image
        sample['thin_image'] = self.thinimage
        sample['endpoints'] = self.endpoints
        sample['selfintersectionpoints'] = self.selfintersectionpoints
        sample['intersectionpoints'] = self.intersectionpoints
        sample['sharppoints'] = self.sharppoints
        with open(filepath, "wb") as fl:
            np.savez_compressed(fl, **sample)

    def clean_white_labels(self):

        self.endpoints = np.array(filter_points_on_white(self.image, self.endpoints))
        self.sharppoints = np.array(filter_points_on_white(self.image, self.sharppoints))
        self.intersectionpoints = np.array(filter_points_on_white(self.image, self.intersectionpoints))
        self.sharppoints = np.array(filter_points_on_white(self.image, self.sharppoints))

    def dump_pts(self, filepath):
        total_poitns = len(self.endpoints) + len(self.sharppoints) + len(self.intersectionpoints) + len(self.selfintersectionpoints)
        with open(filepath, "w") as f:
            f.write(f"{total_poitns} \n")
            for coo in self.endpoints:
                f.write(f"{int(coo[1])} {int(coo[0])} 1\n")
            for coo in self.selfintersectionpoints:
                f.write(f"{int(coo[1])} {int(coo[0])} 3\n")
            for coo in self.intersectionpoints:
                f.write(f"{int(coo[1])} {int(coo[0])} 3\n")
            for coo in self.sharppoints:
                f.write(f"{int(coo[1])} {int(coo[0])} 5\n")


def dump_fromcleaned_svg(items_to_write=10):
    for idx in range(items_to_write):
        print(f"write svg {idx}")
        drawing = Drawing.from_svg(f"../creativebirds_labelme_1000/svg_cleaned/test_{idx}.svg")
        drawing.clean_endpoints()
        thisitem = DatasetItem.from_drawing(drawing)
        # item.update_image(pngpath=f"../creativebirds_labelme/defaultpng/test_{idx}.png")
        thisitem.filter_points()
        thisitem.filter_points_too_close()
        thisitem.dump_npz(f"../creativebirds_labelme_1000/dataset_cleaned/test_{idx}.npz")
        thisitem.dump_json(f"../creativebirds_labelme_1000/json_cleaned/test_{idx}.json", imagePath=f"test_{idx}.png")
        drawing.write_svg(f"../creativebirds_labelme_1000/svg_cleaned_auto/test_{idx}.svg")

def dump_item(ndjsonitem, idx):
    draw = Drawing.from_drawing_data(ndjsonitem['drawing'], raw_ndjson=True, apply_rdp=False)
    draw.clean_endpoints()
    item = DatasetItem.from_drawing(draw)
    item.filter_points()
    item.filter_points_too_close()
    # draw.write_svg(f"../creativebirds_labelme_5000/svg/test_{idx}.svg")
    # item.dump_npz(f"../creativebirds_labelme_5000/dataset/test_{idx}.npz")
    # item.dump_png(f"../creativebirds_labelme_5000/json/test_{idx}.png")
    # item.dump_json(f"../creativebirds_labelme_5000/json/test_{idx}.json", imagePath=f"test_{idx}.png")

    draw.write_svg(f"../creativecreatures_labelme_1000/svg/test_{idx}.svg")
    item.dump_npz(f"../creativecreatures_labelme_1000/dataset/test_{idx}.npz")
    item.dump_png(f"../creativecreatures_labelme_1000/json/test_{idx}.png")
    item.dump_json(f"../creativecreatures_labelme_1000/json/test_{idx}.json", imagePath=f"test_{idx}.png")


def dump_from_ndjson(items_to_write=10):
    # filepath = '/home/ivan/datasets/DOODLERGAN/DOODLERGAN/ndjson/bird_small.ndjson'
    # filepath = '/home/ivan/datasets/DOODLERGAN/DOODLERGAN/ndjson/bird.ndjson'
    filepath = '/home/ivan/datasets/DOODLERGAN/DOODLERGAN/ndjson/creature.ndjson'
    print(f"Reading {filepath}")
    with open(filepath) as f:
        mydata = ndjson.load(f)
        print("Done reading")
    for i in range(items_to_write):
        print(f"write {i}")
        dump_item(mydata[i], idx=i)


def dump_updated(idx):
    item = DatasetItem.from_npz(path_npz=f"../creativebirds_labelme/dataset/test_{idx}.npz")
    item.update_labels_from_json(path_json=f"../creativebirds_labelme/defaultpng/test_{idx}.json")
    item.clean_white_labels()
    item.dump_npz(filepath=f"../creativebirds_labelme/dataset_updated/test_{idx}.npz")


def view_npz(idx):
    item = DatasetItem.from_npz(path_npz=f"../creativebirds_labelme/dataset_updated/test_{idx}.npz")
    # item = DatasetItem.from_npz(path_npz=f"../creativebirds_labelme/dataset/test_{idx}.npz")
    # item.update_labels_from_json(path_json=f"../creativebirds_labelme/defaultpng/test_{idx}.json")
    item.plt_plot()
    plt.show()


def npz_to_pts(idx):
    item = DatasetItem.from_npz(path_npz=f"../creativebirds_labelme/dataset_cleaned/test_{idx}.npz")
    item.dump_pts(filepath=f"../creativebirds_labelme/pts/test_{idx}.png.pts")


if __name__ == "__main__":
    myid = 200
    # item = DatasetItem.from_npz(path_npz=f"../creativebirds_labelme/dataset_updated/test_{myid}.npz")
    # item.update_image(pngpath=f"../creativebirds_labelme/defaultpng/test_102.png")
    filepath = '/home/ivan/datasets/DOODLERGAN/DOODLERGAN/ndjson/bird_small.ndjson'
    print(f"Reading {filepath}")
    with open(filepath) as f:
        mydata = ndjson.load(f)
        print("Done reading")
    # draw = Drawing.from_drawing_data(mydata[myid]['drawing'], raw_ndjson=True, apply_rdp=False)
    # item = DatasetItem.from_drawing(draw)
    # item.filter_points()
    # item.filter_points_too_close()
    # item.plt_plot()
    # plt.legend()
    # plt.show()

    draw = Drawing.from_svg(f"../creativebirds_labelme/svg_cleaned/test_{myid}.svg")
    # draw = Drawing.from_svg(f"../creativebirds_labelme/debug/test_{myid}.svg")
    draw.clean_endpoints()
    item = DatasetItem.from_drawing(draw)
    # item.update_image(pngpath=f"../creativebirds_labelme/defaultpng/test_{myid}.png")
    item.filter_points()
    item.filter_points_too_close()
    item.plt_plot()
    plt.legend()
    plt.show()
    # print(item.image)
    # item.clean_white_labels()
    # item.filter_points()
    # item.plt_plot()
    # plt.legend()
    # plt.show()
    # dump_from_ndjson(1000)
    dump_fromcleaned_svg(1000)
    # for i in range(0,500):
    #     npz_to_pts(i)
    # for i in range(100,200):
    #     dump_updated(i)