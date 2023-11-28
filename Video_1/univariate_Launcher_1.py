from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np


class PointsFromDistribution(Scene):
    def construct(self):
        np.random.seed(0)
        self.camera.background_color = "#0A1622"
        x_axis = NumberLine(x_range=(0, 10), length=10,
                            color=WHITE,
                            include_tip=True,
                            tip_width=0.2,
                            tip_height=0.2,
                            include_numbers=True)

        # Hide the last tick and label
        x_axis.numbers[-1].set_opacity(0)
        x_axis.ticks[-1].set_opacity(0)

        tip_label = MathTex("X").next_to(x_axis.get_end(), DOWN)
        g = VGroup()

        g.add(x_axis, tip_label)
        self.play(Write(g, lag_ratio=0.01, run_time=2))
        for i in range(5):
            sample = get_normal_samples(mu=5, variance=1, n_samples=1)
            rand_point = Circle(radius=0.1, color=BLUE_D,  stroke_width=3.5)
            rand_point.move_to(x_axis.n2p(sample[0]))

            self.play(Write(rand_point), lag_ratio=0.01, run_time=1)


if __name__ == '__main__':

    scene = PointsFromDistribution()
    scene.render()  # That's it!

    # Here is the extra step if you want to also open
    # the movie file in the default video player
    # (there is a little different syntax to open an image)
    open_media_file(scene.renderer.file_writer.movie_file_path)
