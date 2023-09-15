from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
from univariate_Launcher_1 import PointsFromDistribution

class PointsFromDistribution2(Scene):
    def construct(self):
        np.random.seed(0)
        group = VGroup()
        x_axis = self.Launcher_1_scene(group)
        sample = get_normal_samples(mu=5, variance=1, n_samples=50)
        for i in range(50):
            rand_point = Dot(color=RED)
            rand_point.move_to(x_axis.n2p(sample[i]))
            self.play(Write(rand_point),lag_ratio=0.01, run_time=0.1)


    def Launcher_1_scene(self, group):
        x_axis = NumberLine(x_range=(0, 10), length=10,
                            color=BLUE,
                            include_numbers=True)
        group.add(x_axis)
        sample = get_normal_samples(mu=5, variance=1, n_samples=5)
        for i in range(5):
            rand_point = Dot(color=RED)
            rand_point.move_to(x_axis.n2p(sample[i]))
            group.add(rand_point)
        self.add(group)
        return x_axis


if __name__ == '__main__':

    # scene1 = PointsFromDistribution()
    # scene1.render(preview=True)  # That's it!

    scene2 = PointsFromDistribution2()
    scene2.render()

    open_media_file(scene2.renderer.file_writer.movie_file_path)
