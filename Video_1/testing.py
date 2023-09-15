from manim import *

class ChangeAxisRange(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 10],
            y_range=[-1, 1],
            axis_config={"color": BLUE},
        )

        # Add axes to the scene
        self.play(Create(axes))

        # Create a DecimalNumber to represent the new x_range maximum value
        new_x_max = DecimalNumber(10, num_decimal_places=0)
        new_x_max.next_to(axes.x_axis, UP)

        # Animate the change in x_range maximum value
        self.play(ChangeDecimalToValue(new_x_max, 5))

        # Update the x_range
        axes.x_range = [0, 5]
        self.wait(1)