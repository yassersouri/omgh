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

        return parts

