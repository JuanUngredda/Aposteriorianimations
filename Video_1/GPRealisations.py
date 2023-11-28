import matplotlib.pylab as plt
from manim import *
from manim.utils.file_ops import open_file as open_media_file

class GraphExample(Scene):
    CONFIG = {
        "y_min": -5,
        "y_max": 5,
        "x_min": -4,
        "x_max": 4,
        "x_axis_config": {"tick_frequency": 1},
        "y_axis_config": {"tick_frequency": 1}
    }

    def GPRealisationBuilder(self, seed, kernel):

        np.random.seed(seed)
        numberOfXPredictions = 25
        ub = 5
        lb = - 5
        X = np.linspace(lb, ub, numberOfXPredictions)[:, None]
        Z = np.random.normal(loc=0, scale=1, size=numberOfXPredictions + 1).reshape(numberOfXPredictions + 1, 1)

        def gpPriorRealisation(xnew):
            Xextended = np.vstack((np.atleast_2d(xnew), X)).reshape(-1)
            X_entendedSorted = np.sort(Xextended)[:, None]
            xnewIndex = np.argsort(Xextended) == 0
            C = kernel(X_entendedSorted, X_entendedSorted) + np.identity(len(Xextended)) * 1e-9
            L = np.linalg.cholesky(C)
            f = np.dot(L, Z).reshape(-1)
            xnewval = f[xnewIndex][0]
            return xnewval

        return gpPriorRealisation

    def construct(self):
        self.camera.background_color = WHITE

        self.next_section("RBF Kernel")
        rbfgprealisation, kernelMatrix, kerneltext = self.constructscene(KernelName="RBF", inheritedgprealisations=None,
                                                                         kernel_text_object=None,
                                                                         kernelMatrix=None, numberOfGenerations=4)
        self.next_section("Periodic Kernel")
        periodicgprealisation, kernelMatrix, kerneltext = self.constructscene(KernelName="PERIODIC",
                                                                              kernelMatrix=kernelMatrix,
                                                                              kernel_text_object=kerneltext,
                                                                              inheritedgprealisations=rbfgprealisation,
                                                                              numberOfGenerations=4)
        self.next_section("Linear Kernel")
        lineargprealisation, kernelMatrix, kerneltext = self.constructscene(KernelName="LINEAR",
                                                                            kernelMatrix=kernelMatrix,
                                                                            kernel_text_object=kerneltext,
                                                                            inheritedgprealisations=periodicgprealisation,
                                                                            numberOfGenerations=4)
        self.next_section("PeriodicLinear Kernel")
        _, _, _ = self.constructscene(KernelName="PERIODIC+LINEAR", kernelMatrix=kernelMatrix,
                                      inheritedgprealisations=lineargprealisation,
                                      kernel_text_object=kerneltext,
                                      numberOfGenerations=4)

    def constructscene(self, inheritedgprealisations, kernelMatrix, kernel_text_object, KernelName="RBF",
                       numberOfGenerations=5):

        numberOfRealisations = 10
        colorlst = [self.genHexColor(i) for i in range(numberOfRealisations)]

        axes_realisations = Axes(
            x_range=[-3, 3, 1], y_range=[-4, 4, 1], axis_config={"include_tip": False}
        ).set_color(BLACK)
        axes_realisations.shift(RIGHT * -3).scale(0.65)
        x_label = axes_realisations.get_x_axis_label(Tex("$x$", color=BLACK)).shift(DOWN * 0.3)
        y_label = axes_realisations.get_y_axis_label(Tex("$f(x)$", color=BLACK))

        kernelFunction = self.Kernel(type=KernelName)
        oldGPrealisations = [
            axes_realisations.plot(self.GPRealisationBuilder(seed=i, kernel=kernelFunction), color=colorlst[i])
            for i in range(numberOfRealisations)]
        movement_time = 2
        oldkernel = ImageMobject(
            "/home/jungredda/Documents/PythonAnimations/PythonAnimationsThesis/plots/" + KernelName + ".png")
        oldkernel.to_edge(RIGHT * 0.3, buff=4).scale(1.8)

        oldkernel_text_object = self.kernel_text_gen(type=KernelName)
        oldkernel_text_object.to_edge(UP * 0.5, buff=2).scale(1.8)

        if inheritedgprealisations is None:

            self.play(Write(axes_realisations, lag_ratio=0.4, run_time=0.4),  FadeIn(x_label), FadeIn(y_label))

            self.play(
                Create(oldGPrealisations[0]),
                Create(oldGPrealisations[1]),
                Create(oldGPrealisations[2]),
                Create(oldGPrealisations[3]),
                Create(oldGPrealisations[4]),
                Create(oldGPrealisations[5]),
                FadeIn(oldkernel),
                Write(oldkernel_text_object),
                run_time=1.5
            )

        else:

            oldGPrealisations = inheritedgprealisations
            oldkernel = kernelMatrix
            oldkernel_text_object = kernel_text_object



        for j in range(numberOfGenerations):
            newGPrealisations = [
                axes_realisations.plot(
                    self.GPRealisationBuilder(seed=i + numberOfRealisations * j, kernel=kernelFunction),
                    color=colorlst[i]) for i in range(numberOfRealisations)]

            if j == 0 and inheritedgprealisations is not None:
                newkernel = ImageMobject(
                    "/home/jungredda/Documents/PythonAnimations/PythonAnimationsThesis/plots/" + KernelName + ".png")
                newkernel.to_edge(RIGHT * 0.3, buff=4).scale(1.8)

                newkernel_text_object = self.kernel_text_gen(type=KernelName)
                newkernel_text_object.to_edge(UP * 0.5, buff=2).scale(1.8)

                self.play(
                    Transform(oldGPrealisations[0], newGPrealisations[0]),
                    Transform(oldGPrealisations[1], newGPrealisations[1]),
                    Transform(oldGPrealisations[2], newGPrealisations[2]),
                    Transform(oldGPrealisations[3], newGPrealisations[3]),
                    Transform(oldGPrealisations[4], newGPrealisations[4]),
                    Transform(oldGPrealisations[5], newGPrealisations[5]),
                    ReplacementTransform(oldkernel_text_object, newkernel_text_object),
                    FadeOut(oldkernel),
                    FadeIn(newkernel),
                    lag_ratio=0,
                    run_time=movement_time
                )

                self.remove(oldGPrealisations[0],
                            oldGPrealisations[1],
                            oldGPrealisations[2],
                            oldGPrealisations[3],
                            oldGPrealisations[4],
                            oldGPrealisations[5],
                            )

                oldkernel = newkernel
                oldGPrealisations = newGPrealisations
                oldkernel_text_object = newkernel_text_object
            else:
                self.play(
                    Transform(oldGPrealisations[0], newGPrealisations[0]),
                    Transform(oldGPrealisations[1], newGPrealisations[1]),
                    Transform(oldGPrealisations[2], newGPrealisations[2]),
                    Transform(oldGPrealisations[3], newGPrealisations[3]),
                    Transform(oldGPrealisations[4], newGPrealisations[4]),
                    Transform(oldGPrealisations[5], newGPrealisations[5]),
                    lag_ratio=0,
                    run_time=movement_time
                )

                self.remove(oldGPrealisations[0],
                            oldGPrealisations[1],
                            oldGPrealisations[2],
                            oldGPrealisations[3],
                            oldGPrealisations[4],
                            oldGPrealisations[5],

                            )

                oldGPrealisations = newGPrealisations

        return newGPrealisations, oldkernel, oldkernel_text_object

    def kernel_text_gen(self, type):
        if type == "RBF":
            return Tex(r"$k_{SE}(x, x') = \sigma^{2} \exp( -\frac{(x - x')^{2}}{2 l^{2}} )$", font_size=20, color=BLACK)
        elif type == "PERIODIC":
            return Tex(r"$k_{Per}(x, x') = \sigma^{2} \exp( -\frac{2 \sin^{2}(\pi |x - x'| / p)}{l^{2}} )$",
                       font_size=20, color=BLACK)
        elif type == "LINEAR":
            return Tex(r"$k_{Lin}(x, x') = \sigma_{b}^{2} + \sigma_{v}^{2} (x - c)(x' - c)$", font_size=20, color=BLACK)
        elif type == "PERIODIC+LINEAR":
            return Tex(r"$k_{Per + Lin}(x, x') = k_{Per}(x, x') + k_{Lin}(x, x')$", font_size=20, color=BLACK)

    def genHexColor(self, seed):
        np.random.seed(seed)
        r = lambda: np.random.randint(0, 255)
        return format('#%02X%02X%02X' % (r(), r(), r()))

    def Kernel(self, type):
        if type == "RBF":
            return self.RBF_KERNEL
        elif type == "PERIODIC":
            return self.PERIODIC_KERNEL
        elif type == "LINEAR":
            return self.LINEAR_KERNEL
        elif type == "PERIODIC+LINEAR":
            return self.PERIODIC_LINEAR_KERNEL

    def PERIODIC_LINEAR_KERNEL(self, X1, X2):
        return self.LINEAR_KERNEL(X1, X2) + self.PERIODIC_KERNEL(X1, X2)

    def PERIODIC_KERNEL(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        Dim1 = X1.shape[0]
        Dim2 = X2.shape[0]
        kernelMatrix = np.zeros((Dim1, Dim2))
        var = 1.3
        ls = 1.5
        p = 1
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                kernelMatrix[i, j] = var ** 2 * np.exp(- (2 / ls ** 2) * np.sin((np.pi / p) * (x1 - x2)) ** 2)
        return kernelMatrix

    def LINEAR_KERNEL(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        Dim1 = X1.shape[0]
        Dim2 = X2.shape[0]
        kernelMatrix = np.zeros((Dim1, Dim2))
        sigmab = 0.3
        sigmav = 1
        c = 0
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                x1_centered = x1 - c
                x2_centered = x2 - c
                kernelMatrix[i, j] = (x1_centered) * (x2_centered) * sigmav + sigmab

        return kernelMatrix

    def RBF_KERNEL(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        Dim1 = X1.shape[0]
        Dim2 = X2.shape[0]
        kernelMatrix = np.zeros((Dim1, Dim2))
        lengtscale = 0.8
        var = 1
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                kernelMatrix[i, j] = (var ** 2) * np.exp(-(x1 - x2) ** 2 / (2 * (lengtscale ** 2)))

        return kernelMatrix


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

if __name__ == '__main__':
    scene = GraphExample()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)