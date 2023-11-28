from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class StochasticProcesses2(Scene):

    def construct(self):
        group1 = VGroup()
        xy_axis = self.get_xy_axis()
        kernelFunction = self.RBF_KERNEL

        numberOfRealisations = 10
        colorlst = [self.genHexColor(i) for i in range(numberOfRealisations)]

        group1.add(xy_axis)
        for i in range(numberOfRealisations):
            v_objects = self.scatter_plot(xy_axis=xy_axis, fun=self.GPRealisationBuilder(seed=i, kernel=kernelFunction),
                                          x_values=[1, 2], color=colorlst[i])
            group1.add(v_objects)

        self.play(Write(group1[0]))
        for i in range(1, 11):
            self.play(Write(group1[i]))
        self.wait()

        new_xy_axis = self.get_new_xy_axis()
        x_values = np.arange(1, 20)
        group2 = VGroup()
        group2.add(new_xy_axis)
        for i in range(numberOfRealisations):
            v_objects = self.scatter_plot(xy_axis=new_xy_axis,
                                          fun=self.GPRealisationBuilder(seed=i, kernel=kernelFunction),
                                          x_values=x_values[:2], color=colorlst[i])
            group2.add(v_objects)

        self.play(Transform(group1, group2))

        group3 = VGroup()
        group3.add(new_xy_axis)
        for i in range(numberOfRealisations):
            v_objects = self.scatter_plot(xy_axis=new_xy_axis,
                                          fun=self.GPRealisationBuilder(seed=i, kernel=kernelFunction),
                                          x_values=x_values[2:], color=colorlst[i])
            group3.add(v_objects)

        self.play(Create(group3))
        self.wait()

        # newGPrealisations = [
        #     self.scatter_plot(xy_axis=xy_axis,
        #                       fun=self.GPRealisationBuilder(seed=i, kernel=kernelFunction),
        #                       x_values=x_values[2:],
        #                       color=colorlst[i]) for i in range(numberOfRealisations)]

        # n_realisations = 5
        # for i in range(n_realisations):
        #     sample = get_multivaraite_normal_samples(mu=np.ones(len(cov))*5, cov=cov, n_samples=1)
        #     samples.append(sample)
        #     for idx, x in enumerate(x_values[:2]):
        #         rand_point = Dot(xy_axis.coords_to_point(x, sample[idx]), color=RED)
        #         points_group.add(rand_point)
        #
        # self.add(points_group)
        #
        # new_xy_axis = self.get_new_xy_axis()
        # samples = []
        # x_values = np.arange(1,20)
        # cov = [[self.rbf_kernel(x1=i, x2=j) for i in x_values] for j in x_values]
        #
        # group2 = VGroup()
        # group2.add(new_xy_axis)
        # for i in range(n_realisations):
        #     sample = get_multivaraite_normal_samples(mu=np.ones(len(cov))*5, cov=cov, n_samples=1)
        #     samples.append(sample)
        #     for idx, x in enumerate(x_values[:2]):
        #         rand_point = Dot(new_xy_axis.coords_to_point(x, sample[idx]), color=RED)
        #         group2.add(rand_point)
        #
        # self.play(Transform(points_group, group2))
        #
        # samples = []
        # x_values = np.arange(1,20)
        # cov = [[self.rbf_kernel(x1=i, x2=j) for i in x_values] for j in x_values]
        #
        # points_group_3 = VGroup()
        # for i in range(n_realisations):
        #     sample = get_multivaraite_normal_samples(mu=np.ones(len(cov))*5, cov=cov, n_samples=1)
        #     samples.append(sample)
        #     for idx, x in enumerate(x_values[2:]):
        #         rand_point = Dot(new_xy_axis.coords_to_point(x, sample[idx]), color=RED)
        #         points_group_3.add(rand_point)
        # self.play(Write(points_group_3))
        # self.wait()

    def scatter_plot(self, xy_axis, fun, x_values, color):
        g = VMobject()
        for x_val in x_values:
            g.add(Dot(xy_axis.coords_to_point(x_val, fun([x_val])), color=color))
        return g

    def RBF_KERNEL(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        Dim1 = X1.shape[0]
        Dim2 = X2.shape[0]
        kernelMatrix = np.zeros((Dim1, Dim2))
        lengtscale = 3
        var = 2
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                kernelMatrix[i, j] = (var ** 2) * np.exp(-(x1 - x2) ** 2 / (2 * (lengtscale ** 2)))

        return kernelMatrix

    def genHexColor(self, seed):
        np.random.seed(seed)
        r = lambda: np.random.randint(0, 255)
        return format('#%02X%02X%02X' % (r(), r(), r()))

    def rbf_kernel(self, x1, x2, gamma=10):
        squared_distance = np.sum((x1 - x2) ** 2)
        kernel_value = 1 * np.exp(-gamma * squared_distance)
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

        xy_axis = Axes(x_range=(0, 20), y_range=(0, 10), color=BLUE)
        # xy_axis.shift(DOWN * 2)

        return xy_axis

    def GPRealisationBuilder(self, seed, kernel):

        np.random.seed(seed)
        numberOfXPredictions = 25
        ub = 20
        lb = 0
        X = np.linspace(lb, ub, numberOfXPredictions)[:, None]
        Z = np.random.normal(loc=0, scale=1, size=numberOfXPredictions + 1).reshape(numberOfXPredictions + 1, 1)

        def gpPriorRealisation(xnew):
            Xextended = np.vstack((np.atleast_2d(xnew), X)).reshape(-1)
            X_entendedSorted = np.sort(Xextended)[:, None]
            xnewIndex = np.argsort(Xextended) == 0
            C = kernel(X_entendedSorted, X_entendedSorted) + np.identity(len(Xextended)) * 1e-9
            L = np.linalg.cholesky(C)
            f = np.dot(L, Z).reshape(-1)
            xnewval = 5 + f[xnewIndex][0]
            return xnewval

        return gpPriorRealisation


if __name__ == '__main__':
    scene = StochasticProcesses2()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
