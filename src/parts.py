class Part(object):

    def __init__(self, img_id, part_name, part_id, x, y, is_visible):
        self.img_id = img_id
        self.part_name = part_name
        self.part_id = part_id
        self.x = x
        self.y = y
        self.is_visible = is_visible

    def __str__(self):
        return "img_id:%d \tpart_name:%s \tpart_id:%d x:%d \ty:%d \tis_visible:%d" % (self.img_id, self.part_name.ljust(20), self.part_id, self.x, self.y, self.is_visible)

    def __repr__(self):
        return self.__str__()

    def is_part(self, part_name):
        return part_name == self.part_name

    def is_part_id(self, part_id):
        return part_id == self.part_id


class Parts(object):

    def __init__(self, parts):
        self.parts = parts

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return self.parts.__iter__()

    def __str__(self):
        string = '\n'.join([str(p) for p in self])
        return '[' + string + ']'

    def __repr__(self):
        return self.__str__()

    def filter_by_name(self, names):
        filtered_parts = []
        for name in names:
            for part in self.parts:
                if part.is_part(name):
                    filtered_parts.append(part)

        return Parts(filtered_parts)

    def center(self):
        mean_x, mean_y = 0., 0.

        for part in self.parts:
            mean_x += part.x
            mean_y += part.y

        return mean_x/len(self), mean_y/len(self)

    def bounding_width_height(self):
        min_x, max_x, min_y, max_y = 100000, 0, 100000, 0

        for part in self.parts:
            if min_x > part.x:
                min_x = part.x
            if min_y > part.y:
                min_y = part.y

            if max_x < part.x:
                max_x = part.x
            if max_y < part.y:
                max_y = part.y

        return max_x - min_x, max_y - min_y

class CUBParts(object):

    PART_NUMBERS = {'back': 1, 'beak': 2, 'belly': 3, 'breast': 4, 'crown': 5, 'forehead': 6, 'left eye': 7, 'left leg': 8, 'left wing': 9, 'nape': 10, 'right eye': 11, 'right leg': 12, 'right wing': 13, 'tail': 14, 'throat': 15}

    PART_NAMES = {v: k for k, v in PART_NUMBERS.items()}

    DIM_IMG_ID = 0
    DIM_PART_ID = 1
    DIM_PART_X = 2
    DIM_PART_Y = 3
    DIM_VISIBLE = 4

    IS_VISIBLE = 1

    def __init__(self, info, bbox=None):
        self.info = info
        if bbox is not None:
            self.bbox = bbox
        else:
            self.bbox = None

    def for_image(self, img_id):
        """
        returns a list of Part objects
        """
        related_info = self.info[(self.info[:, self.DIM_IMG_ID] == img_id) & (self.info[:, self.DIM_VISIBLE] == self.IS_VISIBLE), :]
        parts = []
        for r_i in related_info:
            img_id = r_i[self.DIM_IMG_ID]
            part_id = r_i[self.DIM_PART_ID]
            part_name = self.PART_NAMES[part_id]
            part_x = r_i[self.DIM_PART_X]
            part_y = r_i[self.DIM_PART_Y]
            is_visible = r_i[self.DIM_VISIBLE]
            parts.append(Part(img_id, part_name, part_id, part_x, part_y, is_visible))

        return Parts(parts)

