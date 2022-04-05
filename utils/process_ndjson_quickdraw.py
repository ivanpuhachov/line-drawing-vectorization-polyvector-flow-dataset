import ndjson
import numpy as np
import math
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from svgpathtools import Line, Path
from svgpathtools import wsvg as svg_write
from utils.process_svg import weighted_line
from skimage.transform import resize
from PIL import Image
# from rdp import rdp
from simplification.cutil import simplify_coords_vw
from skimage.draw import line_aa
from typing import List
from svgpathtools import svg2paths

class SegmentsIntersectionException(Exception):
    def __init__(self, message="Error with segments intersections"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Exception: {self.message}"


class Point2d:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        self.point = Point(x, y)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return (math.fabs(self.x - other.x) < 1e-6) and \
               (math.fabs(self.y - other.y) < 1e-6)

    def to_tuple(self):
        return self.x, self.y

    def to_list(self):
        return [self.x, self.y]

    def shapely(self):
        return self.point

    def from_shapely(self, p: Point):
        self.x = p.x
        self.y = p.y
        self.point = p

    def plt_plot(self):
        plt.scatter(self.x, self.y)


class Segment:
    def __init__(self, point1: Point2d, point2: Point2d):
        self.point1 = point1
        self.point2 = point2
        self.linestring = LineString([point1.shapely(), point2.shapely()])

    def __str__(self):
        return f"|{self.point1}, {self.point2}|"

    def length2(self):
        return (self.point1.x - self.point2.x) ** 2 + \
               (self.point1.y - self.point2.y) ** 2

    def length(self):
        return math.sqrt(self.length2())

    def linestring_reversed(self):
        return LineString([self.point2.shapely(), self.point1.shapely()])

    def to_vector(self):
        return np.array([self.point2.x - self.point1.x, self.point2.y - self.point1.y])

    def cos_angle_with(self, other):
        thisvector = self.to_vector()
        othervector = other.to_vector()
        ab = np.sqrt(np.sum(thisvector**2))
        ac = np.sqrt(np.sum(othervector**2))
        dotprod = np.dot(thisvector, othervector)
        # print(ab, ac, dotprod)
        costheta = dotprod / (ab * ac)
        return costheta

    def is_intersecting_on_endpoints(self, other) -> bool:
        if (self.point1 == other.point1) or \
                (self.point1 == other.point2) or \
                (self.point2 == other.point1) or \
                (self.point2 == other.point2):
            return True
        return False

    def is_intersecting(self, other) -> bool:
        if self.is_intersecting_on_endpoints(other):
            return False
        return self.linestring.intersects(other.linestring)

    def intersect(self, other):
        intersection_point = self.linestring.intersection(other.linestring)
        if intersection_point.is_empty:
            raise SegmentsIntersectionException("Intersection point is empty")
        if type(intersection_point) is Point:
            return Point2d(intersection_point.x, intersection_point.y)
        else:
            # print(intersection_point)
            # if type(intersection_point) is LineString:
            #     return Point2d()
            raise SegmentsIntersectionException("Segments intersect by a line")

    def plt_plot(self, linewidth=3):
        plt.plot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y], c='k', linewidth=linewidth)

    def to_svg_line(self):
        return Line(start=complex(self.point1.x, self.point1.y),
                    end=complex(self.point2.x, self.point2.y))

    def distance(self, point: Point):
        return self.linestring.distance(point)

    def closest_point(self, point: Point):
        p1, p2 = nearest_points(point, self.linestring)
        return p2


class Stroke:
    def __init__(self, linestring):
        self.linestring = linestring
        xStroke, yStroke = linestring.xy
        list_of_segments = []
        for i in range(len(xStroke) - 1):
            list_of_segments.append(
                Segment(
                    Point2d(xStroke[i], yStroke[i]),
                    Point2d(xStroke[i + 1], yStroke[i + 1])
                ))
        self.list_of_segments = list_of_segments

    @classmethod
    def from_strokedata(cls, xStroke, yStroke):
        assert (len(xStroke) == len(yStroke))
        linestring = LineString([(xStroke[i], yStroke[i]) for i in range(len(xStroke))])
        return cls(linestring)

    @classmethod
    def from_drawingdata(cls, drawingdata):
        return cls.from_strokedata(drawingdata[0], drawingdata[1])

    def update_linestring(self, l_segemets: List[Segment]):
        xpoints = [s.point1.x for s in l_segemets]
        xpoints.append(l_segemets[-1].point2.x)
        ypoints = [s.point1.y for s in l_segemets]
        ypoints.append(l_segemets[-1].point2.y)
        linestring = [(xpoints[i], ypoints[i]) for i in range(len(xpoints))]
        self.linestring = LineString(linestring)

    def clean_ends(self):
        list_of_segments = self.list_of_segments
        cos0 = list_of_segments[0].cos_angle_with(list_of_segments[1])
        # print(cos0)
        if np.abs(cos0) < 0.8:
            # print("drop")
            list_of_segments.pop(0)
        cos1 = list_of_segments[-1].cos_angle_with(list_of_segments[-2])
        if np.abs(cos1) < 0.8:
            # print("drop")
            list_of_segments.pop(-1)
        self.list_of_segments = list_of_segments
        self.update_linestring(l_segemets=list_of_segments)

    def distance(self, point: Point):
        segments_dist = [s.linestring.distance(point) for s in self.list_of_segments]
        return min(segments_dist)

    def closest_segment_id(self, point: Point):
        segments_dist = [s.linestring.distance(point) for s in self.list_of_segments]
        mindist = min(segments_dist)
        closest_segment_idx = segments_dist.index(mindist)
        return closest_segment_idx

    def closest_segment(self, point: Point):
        closest_segment_idx = self.closest_segment_id(point)
        return self.list_of_segments[closest_segment_idx]

    def closest_point(self, point: Point):
        closest_segment = self.closest_segment(point)
        p1, p2 = nearest_points(point, closest_segment.linestring)
        return p2

    def get_segments(self):
        xStroke, yStroke = self.linestring.xy
        list_of_segments = []
        for i in range(len(xStroke) - 1):
            list_of_segments.append(
                Segment(
                    Point2d(xStroke[i], yStroke[i]),
                    Point2d(xStroke[i + 1], yStroke[i + 1])
                ))
        return list_of_segments

    def has_selfintersections(self):
        return not self.linestring.is_simple

    def get_selfintersections(self):
        return self.get_filtered_selfintersections()
        # list_of_segments = self.list_of_segments
        # list_of_intersection_points = []
        # for i in range(len(list_of_segments) - 2):
        #     for j in range(i + 2, len(list_of_segments)):
        #         if list_of_segments[i].is_intersecting(list_of_segments[j]):
        #             try:
        #                 list_of_intersection_points.append(
        #                     list_of_segments[i].intersect(list_of_segments[j]).to_list()
        #                 )
        #             except SegmentsIntersectionException as e:
        #                 # TODO: if intersection is line segment - append its midpoint
        #                 pass
        #                 # print("segment self-intersection error occured", str(e))
        # return list_of_intersection_points

    def get_filtered_selfintersections(self):
        """Checks if intersecting segments are almost parallel - in that case, do not add intersection point"""
        list_of_segments = self.list_of_segments
        list_of_intersection_points = []
        for i in range(len(list_of_segments) - 2):
            for j in range(i + 2, len(list_of_segments)):
                if list_of_segments[i].is_intersecting(list_of_segments[j]):
                    cosangle = list_of_segments[i].cos_angle_with(list_of_segments[j])
                    # print(cosangle)
                    if np.abs(cosangle) < 0.8:
                        try:
                            list_of_intersection_points.append(
                                list_of_segments[i].intersect(list_of_segments[j]).to_list()
                            )
                        except SegmentsIntersectionException as e:
                            # TODO: if intersection is line segment - append its midpoint
                            pass
                            # print("segment self-intersection error occured", str(e))
                    else:
                        pass
        return list_of_intersection_points

    def get_endpoints(self, collapse_near_endpoints=True):
        xcoords, ycoords = self.linestring.xy
        if collapse_near_endpoints:  # then we add no endpoints in case if they are too close
            if (xcoords[0] - xcoords[-1]) ** 2 + (ycoords[0] - ycoords[-1]) ** 2 <= 9:
                return []
        return [[xcoords[0], ycoords[0]], [xcoords[-1], ycoords[-1]]]

    def is_intersecting(self, other):
        return self.linestring.intersects(other.linestring)

    def get_intersection_points(self, other):
        try:
            intersection = self.linestring.intersection(other.linestring)
        except:
            return []
        points = []
        if intersection.is_empty:
            return points
        if type(intersection) is Point:
            return [[intersection.x, intersection.y]]
        if type(intersection) is MultiPoint:
            for p in intersection.geoms:
                points.append([p.x, p.y])
            return points
        if type(intersection) is GeometryCollection:
            # TODO: if intersection is a linesegment - append its midpoint
            for p in intersection.geoms:
                if type(p) is Point:
                    points.append([p.x, p.y])
            return points
        return points

    def get_filtered_intersection_points(self, other):
        """If intersection is at a small angle, do not add it"""
        points = []
        for segment in self.list_of_segments:
            for othersegment in other.list_of_segments:
                if segment.is_intersecting(othersegment):
                    try:
                        intersection = segment.intersect(othersegment)
                        cosangle = segment.cos_angle_with(othersegment)
                        # print(cosangle)
                        if np.abs(cosangle) < 0.8:
                            points.append([intersection.x, intersection.y])
                    except:
                        pass
        return points

    def get_sharppoints(self, angle_thr=3.1 * math.pi / 4, debug=False):
        if debug:
            plt.figure()
        list_of_segments = self.list_of_segments
        sharppoints = []
        previos_was_marked = False
        for i in range(len(list_of_segments) - 1):
            if debug:
                list_of_segments[i].plt_plot()
            # if debug:
            #     print(f"Segment {i}: ", list_of_segments[i])
            ab2 = list_of_segments[i].length2()
            bc2 = list_of_segments[i + 1].length2()
            if bc2 < 1:
                if debug:
                    print(f"{i+1}: segment too short: ", list_of_segments[i + 1])
                continue
            ac2 = Segment(point1=list_of_segments[i].point1,
                          point2=list_of_segments[i + 1].point2).length2()
            try:
                angle = math.acos((ab2 + bc2 - ac2) / (2 * math.sqrt(ab2 * bc2)))
            except ZeroDivisionError:
                # print("zero")
                continue
            except ValueError:
                # print(f"value error: {ab2 + bc2 - ac2} / sqrt({4 * ab2 * bc2}) ")
                continue
            if angle < angle_thr:
                if previos_was_marked:
                    previos_was_marked = False
                    if debug:
                        plt.scatter(list_of_segments[i].point2.x, list_of_segments[i].point2.y, c='green')
                else:
                    sharppoints.append(list_of_segments[i].point2.to_list())
                    previos_was_marked = True
                    if debug:
                        # print(f"adding, because angle is {angle}")
                        plt.scatter(list_of_segments[i].point2.x, list_of_segments[i].point2.y, c='red')
                        plt.annotate("{:2.2f}".format(angle),
                                     (list_of_segments[i].point2.x, list_of_segments[i].point2.y))
            else:
                previos_was_marked = False
                if debug:
                    plt.scatter(list_of_segments[i].point2.x, list_of_segments[i].point2.y, c='blue')
                    plt.annotate("{:2.2f}".format(angle), (list_of_segments[i].point2.x, list_of_segments[i].point2.y))
        if debug:
            list_of_segments[-1].plt_plot()
            list_of_segments[0].point1.plt_plot()
            list_of_segments[-1].point2.plt_plot()
            plt.axis('equal')
            plt.show()
        return sharppoints

    def plt_plot(self, linewidth=3):
        list_of_segments = self.list_of_segments
        for seg in list_of_segments:
            seg.plt_plot(linewidth)

    def to_svg_path(self):
        list_of_svg_lines = [segment.to_svg_line() for segment in self.list_of_segments]
        return Path(*list_of_svg_lines)


def apply_rdp_to_strokedata(strokedata):
    a = np.ascontiguousarray(np.array(strokedata).transpose(), dtype=np.float32)
    # return rdp(a, epsilon=1).transpose().tolist()
    return simplify_coords_vw(a, epsilon=40).transpose().tolist()


class Drawing:
    def __init__(self, rawdata: list, list_of_strokes: List[Stroke]):
        self.list_of_strokes = list_of_strokes
        self.rawdata = [[s[0], s[1]] for s in rawdata]  # this is to remove timestamp data from raw ndjson
        self.original_side = 288.0

    @classmethod
    def from_svg(cls, svgpath, canvassize=288):
        paths, attributes = svg2paths(svgpath)
        rawdata = list()
        list_of_stroke = list()
        for path in paths:
            allx = [line.start.real for line in path]
            allx.append(path[-1].end.real)
            ally = [line.start.imag for line in path]
            ally.append(path[-1].end.imag)
            rawdata.append([allx, ally])
            stroke = Stroke.from_drawingdata([allx, ally])
            list_of_stroke.append(stroke)
        return cls(rawdata=rawdata, list_of_strokes=list_of_stroke)

    @classmethod
    def from_drawing_data(cls, drawingdata, raw_ndjson=False, apply_rdp=True, pad=True):
        list_of_strokes = []
        if raw_ndjson:
            # if drawing data is raw do translation and scaling to 256x256 by moving coordinates to range [0,255]
            min_x = np.min([np.min(s[0]) for s in drawingdata])
            min_y = np.min([np.min(s[1]) for s in drawingdata])
            max_x = np.max([np.max(s[0]) for s in drawingdata]) - min_x
            max_y = np.max([np.max(s[1]) for s in drawingdata]) - min_y
            scaling = 255.0 / max(max_x, max_y)

            def rescale_stroke(stroke, tx=min_x, ty=min_y, sc=scaling):
                return [
                    [(x - tx) * sc for x in stroke[0]],
                    [(y - ty) * sc for y in stroke[1]]
                ]

            drawingdata = [rescale_stroke(stroke) for stroke in drawingdata]
            drawingdata = [s for s in drawingdata if len(s[0]) > 1]
        if pad:
            max_x = np.max([np.max(s[0]) for s in drawingdata])
            max_y = np.max([np.max(s[1]) for s in drawingdata])
            pad_x = 16.0 + (255.0 - max_x) / 2
            pad_y = 16.0 + (255.0 - max_y) / 2

            def pad_stroke(stroke, px=pad_x, py=pad_y):
                return [
                    [x + px for x in stroke[0]],
                    [y + py for y in stroke[1]]
                ]

            drawingdata = [pad_stroke(stroke) for stroke in drawingdata]

        for stroke_data in drawingdata:
            processed_stroke = apply_rdp_to_strokedata(stroke_data) if apply_rdp else stroke_data
            newstroke = Stroke.from_drawingdata(processed_stroke)
            list_of_strokes.append(newstroke)
        return cls(drawingdata, list_of_strokes)

    def clean_endpoints(self):
        self.list_of_strokes = [s for s in self.list_of_strokes if len(s.list_of_segments)>3]
        for i in range(len(self.list_of_strokes)):
            self.list_of_strokes[i].clean_ends()

    def get_endpoints(self):
        endpoints = []
        for stroke in self.list_of_strokes:
            endpoints.extend(stroke.get_endpoints())
        return endpoints

    def get_filtered_endpoints(self, close_ditance_thr=3):
        """
        If endpoint lies close to the other endpoint, convert them into sharp corner.
        If an endpoint lies close to the line segment, convert it into intersection
        """
        endpoints_by_segment = [stroke.get_endpoints(collapse_near_endpoints=False) for stroke in self.list_of_strokes]
        new_endpoints = list() #endpoints_by_segment[0]
        new_sharppoints = list()
        new_intersections = list()
        for i_stroke in range(0,len(self.list_of_strokes)):
            for i_point in range(2):
                point = endpoints_by_segment[i_stroke][i_point]
                if i_point == 0:
                    currentsegment = self.list_of_strokes[i_stroke].list_of_segments[0]
                else:
                    currentsegment = self.list_of_strokes[i_stroke].list_of_segments[-1]
                mypoint = Point(point[0],point[1])
                # distances_to_endpoints = [mypoint.distance(Point(otherpoint[0],otherpoint[1])) for otherpoint in new_endpoints]
                # distances_to_endpoints.append(10*close_ditance_thr)
                # if min(distances_to_endpoints) < close_ditance_thr:
                #     closest_endpoint_idx = distances_to_endpoints.index(min(distances_to_endpoints))
                #     otherpoint = new_endpoints.pop(closest_endpoint_idx)
                #     middlepoint = [0.5*(point[0]+otherpoint[0]), 0.5*(point[1]+otherpoint[1])]
                #     new_sharppoints.append(middlepoint)
                #     continue
                othersegments = [self.list_of_strokes[i_stroke].list_of_segments[i] for i in
                                 range(2, len(self.list_of_strokes[i_stroke].list_of_segments) - 2)]
                distances_to_myself = [s.distance(mypoint) for s in othersegments]
                # distances_to_myself = [self.list_of_strokes[i_stroke].list_of_segments[i].distance(mypoint) for i in range(2, len(self.list_of_strokes[i_stroke].list_of_segments)-2)]
                if len(distances_to_myself) > 0:
                    if min(distances_to_myself) < 1.5:
                        mindistidx = distances_to_myself.index(min(distances_to_myself))
                        closest_segment = othersegments[mindistidx]
                        closestpoint = closest_segment.closest_point(mypoint)
                        new_intersections.append([closestpoint.x, closestpoint.y])
                        continue
                otherstrokes = [self.list_of_strokes[j_stroke] for j_stroke in range(len(self.list_of_strokes)) if j_stroke!=i_stroke]
                distances_to_strokes = [s.distance(mypoint) for s in otherstrokes]
                # distances_to_strokes = [self.list_of_strokes[j_stroke].distance(mypoint) for j_stroke in range(len(self.list_of_strokes)) if j_stroke!=i_stroke]
                distances_to_strokes.append(10*close_ditance_thr)
                mindist = min(distances_to_strokes)
                # print(mindist)
                if mindist < close_ditance_thr:
                    mindistidx = distances_to_strokes.index(mindist)
                    closest_stroke = otherstrokes[mindistidx]
                    closest_segment_id = closest_stroke.closest_segment_id(point=mypoint)
                    closest_segment = closest_stroke.list_of_segments[closest_segment_id]
                    cosangle = currentsegment.cos_angle_with(closest_segment)
                    # print(cosangle)
                    # print("append", point)
                    closestpoint = closest_stroke.closest_point(mypoint)
                    # new_intersections.append(point)
                    if np.abs(cosangle)>0.8:
                        # new_endpoints.append([closestpoint.x, closestpoint.y])
                        if (closest_segment_id < 2) and (
                                closest_segment_id >= len(closest_stroke.list_of_segments) - 2):
                            new_endpoints.append([closestpoint.x, closestpoint.y])
                            # pass
                    else:
                        # new_intersections.append([closestpoint.x, closestpoint.y])
                        if (closest_segment_id!=0) and (closest_segment_id!=len(closest_stroke.list_of_segments)-1):

                            new_intersections.append([closestpoint.x, closestpoint.y])
                            # print("new_intersections: ", new_intersections)
                        else:
                            new_sharppoints.append([closestpoint.x, closestpoint.y])
                else:
                    new_endpoints.append(point)
                    pass
        return new_endpoints, new_sharppoints, new_intersections

    def get_filtered_sharp(self, angle_thr=3.1 * math.pi / 4, debug=False):
        sharppoints = []
        endpoints = []
        for stroke in self.list_of_strokes:
            list_of_segments = stroke.list_of_segments
            previos_was_marked = False
            for i in range(len(list_of_segments) - 2):
                distance_to_go = 1
                nextsegment_id = i + 1
                intersectionpoint = list_of_segments[i].point2.shapely()
                currentsegment_length = list_of_segments[i].length()
                nextsegment_length = list_of_segments[nextsegment_id].length()
                if currentsegment_length < distance_to_go:
                    if i>0:
                        point1 = list_of_segments[i-1].linestring.interpolate(-distance_to_go+currentsegment_length)
                    else:
                        point1 = list_of_segments[i].linestring.interpolate(-distance_to_go)
                else:
                    point1 = list_of_segments[i].linestring.interpolate(-distance_to_go)
                if nextsegment_length < distance_to_go:
                    distance_to_go -= nextsegment_length
                    nextsegment_id = i+2
                    nextsegment_length = list_of_segments[nextsegment_id].length()
                    if nextsegment_length >= distance_to_go:
                        point2 = list_of_segments[nextsegment_id].linestring.interpolate(distance_to_go)
                    else:
                        if i + 3 < len(list_of_segments):
                            distance_to_go -= nextsegment_length
                            nextsegment_id = i + 3
                            point2 = list_of_segments[nextsegment_id].linestring.interpolate(distance_to_go)
                        else:
                            point2 = list_of_segments[nextsegment_id].point2.point
                else:
                    point2 = list_of_segments[nextsegment_id].linestring.interpolate(distance_to_go)

                ab2 = intersectionpoint.distance(point1)**2
                bc2 = intersectionpoint.distance(point2)**2
                ac2 = point1.distance(point2)**2
                # ac2 = Segment(point1=list_of_segments[i].point1,
                #               point2=list_of_segments[nextsegment_id].point2).length2()
                try:
                    cosangle = (ab2 + bc2 - ac2) / (2 * math.sqrt(ab2 * bc2))
                    # print(np.arccos(cosangle), " <> ", angle_thr)
                except ZeroDivisionError:
                    # print("zero")
                    continue
                except ValueError:
                    # print(f"value error: {ab2 + bc2 - ac2} / sqrt({4 * ab2 * bc2}) ")
                    continue
                if cosangle > np.cos(angle_thr):
                    # print(ab2, bc2, ac2)
                    # if previos_was_marked:
                    #     previos_was_marked = False
                    # else:
                    #     previos_was_marked = True
                    #     if np.abs(cosangle) < np.abs(np.cos(3.8 * math.pi / 4)):
                    #         sharppoints.append(list_of_segments[i].point2.to_list())
                    #     else:
                    #         endpoints.append(list_of_segments[i].point2.to_list())
                    if cosangle < np.cos(3 * math.pi / 12):
                        # print("sharp")
                        sharppoints.append(list_of_segments[i].point2.to_list())
                    else:
                        # print("here")
                        sharppoints.append(list_of_segments[i].point2.to_list())
                        # endpoints.append(list_of_segments[i].point2.to_list())
                else:
                    previos_was_marked = False
        return endpoints, sharppoints

    def get_selfintersections(self):
        selfintersections = []
        for stroke in self.list_of_strokes:
            if stroke.has_selfintersections():
                selfintersections.extend(stroke.get_filtered_selfintersections())
        return selfintersections

    def get_intersections(self):
        intersections = []
        for i_stroke in range(len(self.list_of_strokes) - 1):
            for j_stroke in range(i_stroke, len(self.list_of_strokes)):
                if self.list_of_strokes[i_stroke].is_intersecting(self.list_of_strokes[j_stroke]):
                    intersections.extend(
                        self.list_of_strokes[i_stroke].get_filtered_intersection_points(self.list_of_strokes[j_stroke])
                    )

        return intersections

    def get_grouped_keypoints(self):
        endpoints = []
        selfintersections = []
        intersections = []
        sharppoints = []
        for i_stroke in range(len(self.list_of_strokes)):
        # for i_stroke in range(len(self.list_of_strokes) - 1):
            stroke = self.list_of_strokes[i_stroke]
            endpoints.extend(stroke.get_endpoints())
            sharppoints.extend(stroke.get_sharppoints())
            if stroke.has_selfintersections():
                selfintersections.extend(stroke.get_selfintersections())
            if i_stroke < len(self.list_of_strokes)-1:
                for j_stroke in range(i_stroke+1, len(self.list_of_strokes)):
                    if self.list_of_strokes[i_stroke].is_intersecting(self.list_of_strokes[j_stroke]):
                        intersections.extend(
                            self.list_of_strokes[i_stroke].get_filtered_intersection_points(self.list_of_strokes[j_stroke])
                        )
        return {"endpoints": np.array(endpoints),
                "selfintersectionpoints": np.array(selfintersections),
                "intersectionpoints": np.array(intersections),
                "sharppoints": np.array(sharppoints)}

    def get_all_keypoints(self):
        keypoints = []
        kp = self.get_grouped_keypoints()
        for key in kp.keys():
            keypoints.extend(list(kp[key]))
        return np.array(keypoints)

    def plt_plot(self, linewidth=3):
        for stroke in self.list_of_strokes:
            stroke.plt_plot(linewidth=linewidth)

    def plt_plot_with_keypoints(self):
        for stroke in self.list_of_strokes:
            stroke.plt_plot()
        kp = self.get_all_keypoints()
        plt.scatter(kp[:, 0], kp[:, 1])

    def render_image(self, side=128, line_diameter=4,
                     return_keypoints=True):
        big_image = np.zeros((int(self.original_side), int(self.original_side)), dtype=np.float)
        # draw on big_image with no padding
        for s in self.rawdata:
            for i in range(len(s[0])-1):
                x1, y1 = s[0][i], s[1][i]
                x2, y2 = s[0][i+1], s[1][i+1]
                try:
                    if ((x1-x2)**2 + (y1-y2)**2 > 0.5):
                        # rr, cc, val = weighted_line(y1,x1,y2,x2, w=line_diameter)
                        rr, cc, val = line_aa(int(y1),int(x1),int(y2),int(x2))
                        big_image[rr,cc] = val
                except ValueError:
                    pass
                except IndexError:
                    pass
        raster_image = resize(big_image, (side, side))
        if return_keypoints:
            keypoints_dict = self.get_grouped_keypoints()
            return raster_image, keypoints_dict
        return raster_image

    def to_dataset_item(self, side=128, line_diameter=5, padding=4, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
        image, keypoints_dict = self.render_image(side, line_diameter=line_diameter,
                                                  return_keypoints=True)
        thin_image = self.render_image(side, line_diameter=2,
                                       return_keypoints=False)
        # thin_image = gaussian(thin_image, sigma=0.5)
        item = {"image": image, "thin_image": thin_image, **keypoints_dict}
        return item

    def dataset_dump(self, path, side=128, line_diameter=5):
        item = self.to_dataset_item(side=side, line_diameter=line_diameter)
        with open(path, "wb") as fl:
            # print("Dumping: ", item["endpoints"][0])
            np.savez_compressed(fl, **item)
            # np.savez_compressed(fl, keypoints)

    def to_svg_list(self):
        return [stroke.to_svg_path() for stroke in self.list_of_strokes]

    def write_svg(self, path):
        # TODO: investigate svg attributes and fix Adobe Illustrator issues here
        # print("writing", self.list_of_strokes)
        # print(path)
        # svg_write(self.to_svg_list(), filename=path)
        magick_strokewidth = 0.2537178821563721
        svg_write(self.to_svg_list(), filename=path, #viewbox=(0, 0, 288, 288),
                  stroke_widths=[magick_strokewidth for stroke in self.list_of_strokes],
                  dimensions=("288px", "288px"),
                  svg_attributes={'xmlns': 'http://www.w3.org/2000/svg', 'xmlns:ev': 'http://www.w3.org/2001/xml-events', 'xmlns:xlink': 'http://www.w3.org/1999/xlink', 'baseProfile': 'full', 'height': '288px', 'version': '1.1', 'viewBox': '0 0 288 288', 'width': '288px'}
                  )


def test_loading(data, myindex):
    drawing = data[myindex]['drawing']
    print(drawing)
    draw = Drawing.from_drawing_data(drawing, raw_ndjson=True, apply_rdp=False)

    draw.write_svg("../example_svg/cat_"+str(myindex) + ".svg")

    ends = np.array(draw.get_endpoints())
    # print(ends)
    selfs = np.array(draw.get_selfintersections())
    print("selfs: ", selfs)
    inters = np.array(draw.get_intersections())
    # print(inters)
    # kp = draw.get_all_keypoints()
    # print(kp)
    # plt.figure()
    # draw.plt_plot_with_keypoints()
    # plt.show()

    item = draw.to_dataset_item(side=288)
    plt.figure(figsize=(7, 7))
    plt.imshow(item["image"], cmap='gray_r')
    # draw.plt_plot()
    # canvas = plt.get_current_fig_manager().canvas
    # canvas.draw()
    # pil_image = Image.frombytes('RGB', canvas.get_width_height(),
    #                                 canvas.tostring_rgb())
    # img_array = np.asarray(pil_image)
    # plt.imshow(img_array, cmap='gray_r')
    for key in item.keys():
        if key.endswith("points"):
            if len(item[key]) > 0:
                plt.scatter(item[key][:, 0], item[key][:, 1], label=key, alpha=0.7, marker='x', linewidths=1)
    # print(keyp)
    plt.title(f"index: {myindex}")
    plt.legend()
    plt.axis('off')
    plt.show()


def png_to_numpy(image_path, side=None):
    im_frame = Image.open(image_path).convert('L')
    if not (side is None):
        crop_size = 9
        im_frame = im_frame.crop(
            (crop_size, crop_size, im_frame.size[0] - crop_size - 2, im_frame.size[1] - crop_size - 2))
        im_frame = im_frame.resize(size=(side, side))
    np_frame = np.array(im_frame.getdata()).astype(np.float32) / 256.0
    np_frame = 1 - np_frame
    np_frame = np.resize(np_frame, (im_frame.size[1], im_frame.size[0]))
    return np_frame


def test2(data, myindex):
    drawing = data[myindex]['drawing']
    draw = Drawing.from_drawing_data(drawing, raw_ndjson=False)
    draw.write_svg(f"old_rabbit_{myindex}.svg")
    draw.dataset_dump(f"old_rabbit_{myindex}.npz")


def debugging():
    fake_data = [
        [
            [0, 255],
            [200, 10]
        ],
        [
            [20, 20],
            [0, 200]
        ]
    ]
    draw = Drawing.from_drawing_data(fake_data, raw_ndjson=True)
    # draw.write_svg("fakesvg.svg")

    ends = np.array(draw.get_endpoints())
    print(ends)
    selfs = np.array(draw.get_selfintersections())
    print(selfs)
    inters = np.array(draw.get_intersections())
    print(inters)
    kp = draw.get_all_keypoints()
    print(kp)
    # plt.figure()
    # draw.plt_plot_with_keypoints()
    # plt.show()

    item = draw.to_dataset_item(side=64)
    # im = gaussian(im, sigma=0.5)
    plt.figure(figsize=(7, 7))
    plt.imshow(item["image"], cmap='gray_r')
    # plt.scatter([5],[0])
    for key in item.keys():
        if key.endswith("points"):
            if len(item[key]) > 0:
                plt.scatter(item[key][:, 0], item[key][:, 1], label=key, alpha=0.7)
    # print(keyp)
    plt.legend()
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # filepath = '/home/ivan/datasets/quickdraw_ndjson/cat.ndjson'
    filepath = '/home/ivan/datasets/DOODLERGAN/DOODLERGAN/ndjson/bird_small.ndjson'
    with open(filepath) as f:
        mydata = ndjson.load(f)
        print("Done reading")
    test_loading(mydata, 4)
    # test_loading(mydata, 6)
    # test_loading(mydata, 11)
    # test_loading(mydata, 6113)
    # test_loading(mydata, 128)
    # test_loading(mydata, 256)
    # test_loading(mydata, 314)
    # test_loading(mydata, 271)
    # test_loading(mydata, 2101)
    # im = Image.open("utils/fakesvg_0.png")
    # plt.imshow(im)
    # plt.show()
    # test2(mydata, 1432)
    # test2(mydata, 12)
    # test2(mydata, 123)
    print("done")
