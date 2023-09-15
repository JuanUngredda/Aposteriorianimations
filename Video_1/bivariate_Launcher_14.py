from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution14(Scene):

    def construct(self):
        samples = []
        for i in range(400):
            sample = get_multivaraite_normal_samples(mu=np.ones(2) * 5, cov=np.identity(2), n_samples=1)
            samples.append(sample)

        xy_axis = self.get_xy_axis()
        points_group = VGroup()
        for s in samples:
            rand_point = Dot(xy_axis.coords_to_point(s[0], s[1]), color=RED)
            points_group.add(rand_point)

        g1 = VGroup(xy_axis, points_group)
        self.add(g1)
        self.wait()

        self.create_contour_curves(xy_axis)
        self.wait()

    def create_contour_curves(self, axis):
        x = np.linspace(0, 10, 500)
        y = np.linspace(0, 10, 500)
        X, Y = np.meshgrid(x, y)

        from scipy.stats import multivariate_normal
        import matplotlib.pyplot as plt
        rv = multivariate_normal([5, 5], [[1, 0], [0, 1]])
        data = np.dstack((X, Y))
        z = rv.pdf(data)
        obj_contour = plt.contour(x, y, z, np.array(
            [z.min(), np.percentile(z, 30), np.percentile(z, 70), np.percentile(z, 95), z.max()]))

        g = VGroup()
        for c in obj_contour.collections[1:]:
            for i in c.get_paths():
                coordinates = i.vertices
                coordinates = np.hstack([coordinates, np.zeros((len(coordinates), 1))])
                ellipse_curve = VMobject()
                axis_points = [axis.coords_to_point(x, y, 0) for x, y, _ in coordinates]
                ellipse_curve.set_points_smoothly(axis_points)

                g.add(ellipse_curve)

        self.play(Write(g))

        return 0

    def generate_samples_object(self, n_samples, group, x_axis, seed=0):
        sample = get_normal_samples(n_samples=n_samples, mu=5, variance=1, seed=seed)
        histogram_object = plt.hist(sample, bins=50, range=(0, 10))
        func = x_axis.plot(lambda x: self.histogram_evaluator(x, histogram_object))
        rects_right = x_axis.get_riemann_rectangles(
            func,
            x_range=[0, 10],
            dx=0.25,
            input_sample_type="right",
        )

        return group, rects_right

    def histogram_evaluator(self, x, histogram_object):

        pos = histogram_object[1]
        pos_vals = histogram_object[0]
        pos_vals = pos_vals / sum(pos_vals) * np.min([60, sum(pos_vals)])
        distance = np.abs(x - pos[:-1])
        idx = np.argmin(distance)
        if distance[idx] > 0.2:
            value = 0
        else:
            value = pos_vals[idx]
        return value

    def Launcher_4_scene(self, group):
        np.random.seed(0)
        x_axis = Axes(x_range=(0, 10), color=BLUE, y_axis_config={"color": None})
        x_axis.shift(DOWN * 2)

        return x_axis

    def get_xy_axis(self):

        xy_axis = Axes(x_range=(0, 10), y_range=(0, 10), color=BLUE)
        # xy_axis.shift(DOWN * 2)

        return xy_axis


if __name__ == '__main__':
    scene = PointsFromDistribution14()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
