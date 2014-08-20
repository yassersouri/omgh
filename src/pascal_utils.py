import re


def which_one(str, arr):
    for a in arr:
        if a in str:
            return a
    return ''


class VOC2006AnnotationParser(object):
    SKIP_CHARACTER = '#'
    OBJECT_SUMMARY = 'Objects with ground truth'
    PREPEND = 'PAS'
    TRUNC = 'Trunc'
    DIFFICULT = 'Difficult'
    CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'cat', 'cow', 'dog', 'horse', 'sheep', 'person']
    VIEWS = ['Frontal', 'Rear', 'Left', 'Right']
    RE_OBJECT_DEF = r"Original label for object (\d+) \"(\S+)\" : \"(\S+)\""
    RE_OBJECT_BB = r"Bounding box for object %d \"%s\" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)"

    def __init__(self, annotation_file_contet):
        self.annotation_file_contet = annotation_file_contet

    def get_objects(self, trunc=True, difficult=False):
        objects = []
        for match in re.finditer(self.RE_OBJECT_DEF, self.annotation_file_contet):
            obj_index, obj_label, original_obj_label = match.groups()
            obj_index = int(obj_index)
            xmin, ymin, xmax, ymax = re.search(self.RE_OBJECT_BB % (obj_index, obj_label), self.annotation_file_contet).groups()
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            if not trunc and self.TRUNC in original_obj_label:
                continue
            if not difficult and self.DIFFICULT in original_obj_label:
                continue
            objects.append({'ind': obj_index, 'label': obj_label, 'original_label': original_obj_label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'trunc': self.TRUNC in original_obj_label, 'difficult': self.DIFFICULT in original_obj_label, 'class': which_one(original_obj_label, self.CLASSES), 'view': which_one(original_obj_label, self.VIEWS)})

        return objects
