from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np


class PointsFromDistribution(Scene):
    def construct(self):
        np.random.seed(0)
        x_axis = NumberLine(x_range=(0, 10), length=10,
                            color=BLUE,
                            include_numbers=True)

        self.play(Write(x_axis, lag_ratio=0.01, run_time=1))

        for i in range(5):
            sample = get_normal_samples(mu=5, variance=1, n_samples=1)
            rand_point = Dot(color=RED)
            rand_point.move_to(x_axis.n2p(sample[0]))

            self.play(Write(rand_point), lag_ratio=0.01, run_time=1)


if __name__ == '__main__':
    scene = PointsFromDistribution()
    scene.render()  # That's it!

    # Here is the extra step if you want to also open
    # the movie file in the default video player
    # (there is a little different syntax to open an image)
    open_media_file(scene.renderer.file_writer.movie_file_path)
