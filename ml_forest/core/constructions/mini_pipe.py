from ml_forest.core.elements.identity import Base


class MiniPipe(object):
    def flow_to(self, node):
        db = node.pipe_init.db
        filepaths = node.pipe_init.filepaths

        # TODO: this part should be replaced by a real training/transforming process
        tmp = Base(db, filepaths)
        tmp.save_db_file()
        ########################################

        node.obj_id = tmp.obj_id
