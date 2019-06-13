import numpy as np


class MoveType(object):
    MISSING = -1
    MOVED = 0
    NONE = 1


class Comparator(object):

    def __init__(self, detector):
        self.detector = detector
        self.eps = 0.05

    def find_close(self, previous_marked_object, actual_objects):
        w, h = self.detector.get_frame_shape()
        status = MoveType.MISSING

        for obj in actual_objects:
            # if previous_marked_object.category and obj.category:
            #     print("{} {}".format(previous_marked_object.category, obj.category))
            if previous_marked_object.category == 'device' and obj.category == 'human':
                _, _, d = previous_marked_object.get_euclidean_norms(obj)
                if d/w < 0.2:
                    input("Person using dev")


        same_class_objs = [(o, o.get_euclidean_norms(previous_marked_object)) \
                           for o in actual_objects \
                           if o.class_id == previous_marked_object.class_id]
        # or o.category == previous_marked_object.category

        if not same_class_objs:
            return MoveType.MISSING
        else:
            i = np.argmin(map(lambda x: x[1][2], same_class_objs))
            now_marked_object, (dx, dy, dxy) = same_class_objs[i]
        # for now_marked_object, (dx, dy, dxy) in same_class_objs:
            if now_marked_object.class_id == previous_marked_object.class_id or\
                    now_marked_object.category == previous_marked_object.category:
                status = MoveType.MOVED if dx/w > self.eps or dy/h > self.eps else MoveType.NONE

        return status

    def compare_objects_lists(self, previous, actual):
        # print('Comparing changes in detected object locations')
        if len(previous) > len(actual):
            return "Some objects has been moved out of sight"
        moved = 0
        for marked_object in previous:
            finding_result = self.find_close(marked_object, actual)
            self.draw_frame(marked_object, False or finding_result != MoveType.NONE)
            if finding_result == MoveType.MOVED:
                moved += 1
            elif finding_result == MoveType.MISSING:
                print("Number of object has changed")
            if moved == 3:
                print("Warning")

    def draw_frame(self, obj, ismoving):
        x1, y1, x2, y2 = map(int, obj.get_cords())
        self.detector.draw_enclosing_frame(obj.class_id, x1, y1, x2, y2, ismoving)

