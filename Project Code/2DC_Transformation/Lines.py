# This routines are used to draw different types of lines on a pfd.
# They are based on the pdf library in python3
def add_double_line(pdf, spacing, gap, start_pos, x_offset, color=None, color2=None):
    """Draw two parallel lines with optional different colors"""
    current_y = pdf.get_y() + spacing
    original_width = pdf.line_width

    # Store original color by getting current state
    pdf.line(0, 0, 0, 0)  # This forces color state to be set
    original_color_str = pdf.draw_color

    # First line
    if color:
        pdf.set_draw_color(*color)
    pdf.set_line_width(0.2)
    pdf.line(start_pos, current_y, pdf.w - x_offset, current_y)

    # Second line
    if color2:
        pdf.set_draw_color(*color2)
    elif color:
        pdf.set_draw_color(*color)
    pdf.line(start_pos, current_y + gap, pdf.w - x_offset, current_y + gap)

    # Restore settings
    pdf.set_line_width(original_width)
    pdf.draw_color = original_color_str  # Directly restore the color string

    return pdf

def add_dashed_line(pdf, spacing, start_pos, x_offset, dash_length, space_length, color=None):
    """Draw a dashed line with optional color"""
    current_y = pdf.get_y() + spacing
    x1, x2 = start_pos, pdf.w - x_offset
    dash_total = dash_length + space_length

    # Store original color state
    pdf.line(0, 0, 0, 0)  # This forces color state initialization
    original_color_str = pdf.draw_color

    if color:
        pdf.set_draw_color(*color)

    # Draw dashed line
    for i in range(int((x2 - x1) / dash_total)):
        pdf.line(
            x1 + (i * dash_total),
            current_y,
            x1 + (i * dash_total) + dash_length,
            current_y
        )

    # Restore original color
    pdf.draw_color = original_color_str

    return pdf


def add_thin_think_line(pdf, spacing, line_width, start_pos, x_offset, color=None):
    """Draw a thin or heavy line depending on line_width (0.1mm) with optional color"""
    # Store original settings
    original_width = pdf.line_width
    pdf.line(0, 0, 0, 0)  # Initialize color state
    original_color_str = pdf.draw_color

    # Set thin line and color
    pdf.set_line_width(line_width)
    if color:
        pdf.set_draw_color(*color)

    # Draw the line
    current_y = pdf.get_y() + spacing
    pdf.line(start_pos, current_y, pdf.w - x_offset, current_y)

    # Restore original settings
    pdf.set_line_width(original_width)
    pdf.draw_color = original_color_str  # Direct string assignment

    return pdf