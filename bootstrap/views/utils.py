

class MultiViews():

    def __init__(self, views):
        super(MultiViews, self).__init__()
        self.views = views

    def generate(self):
        for v in self.views:
            v.generate()
