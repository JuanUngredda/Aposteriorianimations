from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class StochasticProcesses1(Scene):

    def construct(self):
        xy_axis = self.get_xy_axis()
        self.add(xy_axis)

        samples = []
        x_values = np.arange(1,20)
        cov = [[self.rbf_kernel(x1=i, x2=j) for i in x_values] for j in x_values]

        for i in range(10):
            sample = get_multivaraite_normal_samples(mu=np.ones(len(cov))*5, cov=cov, n_samples=1)
            samples.append(sample)
            points_group = VGroup()
            for idx, x in enumerate(x_values[:2]):
                rand_point = Dot(xy_axis.coords_to_point(x, sample[idx]), color=RED)
                points_group.add(rand_point)
            self.play(Write(points_group))


    def rbf_kernel(self, x1, x2, gamma=10):
        squared_distance = np.sum((x1 - x2) ** 2)
        kernel_value = 0.2 * np.exp(-gamma * squared_distance)
        return kernel_value

    def create_contour_curves(self, axis, cov):
        x = np.linspace(0, 10, 500)
        y = np.linspace(0, 10, 500)
        X, Y = np.meshgrid(x, y)

        from scipy.stats import multivariate_normal
        import matplotlib.pyplot as plt
        rv = multivariate_normal([5, 5], cov)
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

        return g

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

        xy_axis = Axes(x_range=(0, 3), y_range=(0, 10), color=BLUE)
        # xy_axis.shift(DOWN * 2)

        return xy_axis

    def get_new_xy_axis(self):

        xy_axis = Axes(x_range=(0, 10), y_range=(0, 10), color=BLUE)
        # xy_axis.shift(DOWN * 2)

        return xy_axis

if __name__ == '__main__':
    scene = StochasticProcesses1()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
