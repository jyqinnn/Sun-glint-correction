class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'sun_glint':
            return 'F:/coral_segmentation/datasets/sun_glint/data'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError