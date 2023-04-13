from typing import Union

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Polygon


def _iou_bbox(box1: BoundingBox, box2: BoundingBox) -> float:
    # get coordinates of the intersection rectangle
    x1_inter = max(box1.top_left[0], box2.top_left[0])
    y1_inter = max(box1.top_left[1], box2.top_left[1])
    x2_inter = min(box1.bottom_right[0], box2.bottom_right[0])
    y2_inter = min(box1.bottom_right[1], box2.bottom_right[1])

    # check if there is an intersection
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    if intersection <= 0:
        return 0.0

    box1_width = box1.bottom_right[0] - box1.top_left[0]
    box1_height = box1.bottom_right[1] - box1.top_left[1]
    box1_area = box1_width * box1_height

    box2_width = box2.bottom_right[0] - box2.top_left[0]
    box2_height = box2.bottom_right[1] - box2.top_left[1]
    box2_area = box2_width * box2_height

    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0.0
    return iou


def iou(a: Union[BoundingBox, Polygon], b: Union[BoundingBox, Polygon]) -> float:
    if isinstance(a, BoundingBox) and isinstance(b, BoundingBox):
        return _iou_bbox(a, b)

    def as_shapely_polygon(obj: Union[BoundingBox, Polygon]) -> ShapelyPolygon:
        if isinstance(obj, BoundingBox):
            (tlx, tly), (brx, bry) = obj.top_left, obj.bottom_right
            return make_valid(ShapelyPolygon([(tlx, tly), (brx, tly), (brx, bry), (tlx, bry)]))
        return make_valid(ShapelyPolygon(obj.points))

    polygon_a = as_shapely_polygon(a)
    polygon_b = as_shapely_polygon(b)
    union = polygon_a.union(polygon_b).area
    return polygon_a.intersection(polygon_b).area / union if union > 0 else 0
