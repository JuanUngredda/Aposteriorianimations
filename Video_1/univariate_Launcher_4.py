from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution4(Scene):

    def construct(self):
        group = VGroup()
        x_axis, points_objects = self.Launcher_2_scene(group)

        old_group, old_rects_right = self.generate_samples_object(n_samples=50,
                                                                  group=group,
                                                                  points_objects=points_objects,
                                                                  x_axis=x_axis)
        self.add(x_axis, old_group, old_rects_right)
        for s in np.array([  50, 500, 5000, 50000]):
            for d in range(5):
                _, new_rects_right = self.generate_samples_object(n_samples=s,
                                                                  group=group,
                                                                  points_objects=points_objects,
                                                                  x_axis=x_axis,
                                                                  seed=d)
                self.play(Transform(old_rects_right, new_rects_right), run_time=2)


    def generate_samples_object(self, n_samples, group, points_objects, x_axis, seed=0):
        sample = get_normal_samples(n_samples=n_samples, mu=5, variance=1, seed=seed)
        histogram_object = plt.hist(sample, bins=50, range=(0, 10))
        func = x_axis.plot(lambda x: self.histogram_evaluator(x, histogram_object))
        rects_right = x_axis.get_riemann_rectangles(
            func,
            x_range=[0, 10],
            dx=0.25,
            input_sample_type="right",
        )

        for obj in points_objects:
            group.remove(obj)
        return group, rects_right

    def histogram_evaluator(self, x, histogram_object):

        pos = histogram_object[1]
        pos_vals = histogram_object[0]
        pos_vals = pos_vals/sum(pos_vals) * np.min([60, sum(pos_vals)])
        distance = np.abs(x - pos[:-1])
        idx = np.argmin(distance)
        if distance[idx] > 0.2:
            value = 0
        else:
            value = pos_vals[idx]
        return value

    def Launcher_2_scene(self, group):
        np.random.seed(0)
        x_axis = Axes(x_range=(0, 10), color=BLUE, y_axis_config={"color": None})
        x_axis.shift(DOWN*2)
        group.add(x_axis)
        sample = get_normal_samples(mu=5, variance=1, n_samples=55)
        points_objects = VGroup()
        for i in range(55):
            rand_point = Dot(color=RED)
            rand_point.move_to(x_axis.c2p(sample[i]))
            group.add(rand_point)
            points_objects.add(rand_point)

        return x_axis, points_objects


if __name__ == '__main__':
    scene = PointsFromDistribution4()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
