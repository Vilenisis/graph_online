"""Interactive function plotting application with GUI controls.

This script uses Tkinter for the interface and Matplotlib for rendering. Users can
enter mathematical expressions evaluated via ``numexpr`` and adjust detected
parameters dynamically. Multiple curves (layers) can be displayed simultaneously,
and options such as logarithmic axes, grid visibility, cursor readouts, and color
customisation are provided. Generated plots can be saved as images, and parameter
settings can be reset quickly.
"""
from __future__ import annotations

import ast
import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence

import matplotlib
import numexpr as ne
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Используем TkAgg для интеграции с Tkinter.
matplotlib.use("TkAgg")


# ``numexpr`` exposes a fairly large collection of functions; collecting the
# allowed names helps us differentiate between callable/function names and real
# parameters that should be exposed to the user.
NUMEXPR_ALLOWED_NAMES = {
    "x",
    "pi",
    "e",
    # Arithmetic helpers
    "where",
    # Trigonometric functions
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    # Exponentials and logarithms
    "exp",
    "log",
    "log10",
    "log1p",
    "expm1",
    # Power helpers
    "sqrt",
    "abs",
    "clip",
    "pow",
    "sign",
    # Rounding
    "ceil",
    "floor",
    "round",
    # Comparison operators
    "maximum",
    "minimum",
}

# ``numexpr`` also allows numpy arrays and scalars. ``math`` constants may also be
# referenced, so we add them to the pool of recognised names to avoid exposing
# them as parameters.
NUMEXPR_ALLOWED_NAMES.update(dir(math))


@dataclass
class CurveDefinition:
    """Stores the information required to render a single curve."""

    label: str
    expression: str
    parameters: Dict[str, float]
    color: str
    x_min: float
    x_max: float
    samples: int = 800

    def evaluate(self) -> Optional[np.ndarray]:
        """Evaluate the expression for plotting.

        Returns ``None`` if the expression cannot be evaluated for the supplied
        settings.
        """

        if self.x_min >= self.x_max:
            raise ValueError("Минимальное значение X должно быть меньше максимального")

        x_values = np.linspace(self.x_min, self.x_max, self.samples)
        local_vars = dict(self.parameters)
        local_vars["x"] = x_values
        local_vars.setdefault("pi", np.pi)
        local_vars.setdefault("e", np.e)

        try:
            result = ne.evaluate(self.expression, local_dict=local_vars)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Не удалось вычислить выражение: {exc}") from exc

        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        return result


class GraphApp:
    """Tkinter-based plotting application."""

    DEFAULT_COLOR = "#1f77b4"

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Интерактивный график функции")
        master.geometry("1100x700")

        # Data storage for plotted curves
        self.curves: List[CurveDefinition] = []

        # Tk variables used across controls
        self.label_var = tk.StringVar()
        self.xmin_var = tk.DoubleVar(value=-10.0)
        self.xmax_var = tk.DoubleVar(value=10.0)
        self.samples_var = tk.IntVar(value=800)
        self.grid_var = tk.BooleanVar(value=True)
        self.logx_var = tk.BooleanVar(value=False)
        self.logy_var = tk.BooleanVar(value=False)
        self.cursor_var = tk.BooleanVar(value=True)
        self.color_var = tk.StringVar(value=self.DEFAULT_COLOR)
        self.axis_x_color_var = tk.StringVar(value="#333333")
        self.axis_y_color_var = tk.StringVar(value="#333333")
        self.cursor_var.trace_add("write", lambda *_: self._toggle_cursor())

        self.param_vars: Dict[str, tk.DoubleVar] = {}
        self.param_frames: Dict[str, tk.Entry] = {}

        self._build_interface()
        self._build_plot()

        # Cursor visual elements
        self.cursor_hline = None
        self.cursor_vline = None
        self.cursor_text = None
        self._connect_cursor_events()

        # Draw initial axes even when there are no curves yet.
        self.replot_all()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_interface(self) -> None:
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        plot_container = ttk.Frame(main_frame)
        plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Expression input and parameter handling
        expr_label = ttk.Label(control_frame, text="Формула (используйте x):")
        expr_label.grid(row=0, column=0, sticky="w")

        self.expr_entry = tk.Text(control_frame, width=40, height=1, undo=True, wrap="none")
        self.expr_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 6))
        self.expr_entry.bind("<FocusOut>", lambda _event: self.update_parameters())
        self.expr_entry.bind("<Return>", lambda _event: "break")
        self.expr_entry.bind("<KP_Enter>", lambda _event: "break")
        self._bind_edit_shortcuts(self.expr_entry)
        self._set_expression("")

        parse_btn = ttk.Button(control_frame, text="Обновить параметры", command=self.update_parameters)
        parse_btn.grid(row=1, column=2, padx=(6, 0))

        reset_btn = ttk.Button(control_frame, text="Сброс параметров", command=self.reset_parameters)
        reset_btn.grid(row=1, column=3, padx=(6, 0))

        param_label = ttk.Label(control_frame, text="Параметры:")
        param_label.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.param_frame = ttk.Frame(control_frame)
        self.param_frame.grid(row=3, column=0, columnspan=4, sticky="we")

        # --- Range and sampling controls
        range_frame = ttk.LabelFrame(control_frame, text="Область определения")
        range_frame.grid(row=4, column=0, columnspan=4, sticky="we", pady=(10, 0))

        ttk.Label(range_frame, text="X min").grid(row=0, column=0, sticky="w")
        ttk.Entry(range_frame, textvariable=self.xmin_var, width=10).grid(row=0, column=1, padx=4)

        ttk.Label(range_frame, text="X max").grid(row=0, column=2, sticky="w")
        ttk.Entry(range_frame, textvariable=self.xmax_var, width=10).grid(row=0, column=3, padx=4)

        ttk.Label(range_frame, text="Точек").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(range_frame, from_=50, to=5000, increment=50, textvariable=self.samples_var, width=8).grid(
            row=1, column=1, padx=4
        )

        # --- Appearance controls
        appearance = ttk.LabelFrame(control_frame, text="Оформление")
        appearance.grid(row=5, column=0, columnspan=4, sticky="we", pady=(10, 0))

        ttk.Checkbutton(appearance, text="Логарифмическая ось X", variable=self.logx_var, command=self.replot_all).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(appearance, text="Логарифмическая ось Y", variable=self.logy_var, command=self.replot_all).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(appearance, text="Показывать сетку", variable=self.grid_var, command=self.replot_all).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Checkbutton(appearance, text="Курсор значений", variable=self.cursor_var).grid(row=3, column=0, sticky="w")

        color_row = ttk.Frame(appearance)
        color_row.grid(row=4, column=0, pady=6, sticky="we")
        ttk.Label(color_row, text="Цвет графика:").pack(side=tk.LEFT)
        self.color_preview = tk.Label(color_row, text="      ", bg=self.color_var.get(), relief=tk.SUNKEN)
        self.color_preview.pack(side=tk.LEFT, padx=4)
        ttk.Button(color_row, text="Выбрать", command=self.choose_color).pack(side=tk.LEFT)

        axis_color_row = ttk.Frame(appearance)
        axis_color_row.grid(row=5, column=0, sticky="we")

        x_axis_row = ttk.Frame(axis_color_row)
        x_axis_row.pack(fill=tk.X, pady=2)
        ttk.Label(x_axis_row, text="Цвет оси X:").pack(side=tk.LEFT)
        self.axis_x_color_preview = tk.Label(x_axis_row, text="      ", bg=self.axis_x_color_var.get(), relief=tk.SUNKEN)
        self.axis_x_color_preview.pack(side=tk.LEFT, padx=4)
        ttk.Button(x_axis_row, text="Выбрать", command=lambda: self.choose_axis_color("x")).pack(side=tk.LEFT)

        y_axis_row = ttk.Frame(axis_color_row)
        y_axis_row.pack(fill=tk.X, pady=2)
        ttk.Label(y_axis_row, text="Цвет оси Y:").pack(side=tk.LEFT)
        self.axis_y_color_preview = tk.Label(y_axis_row, text="      ", bg=self.axis_y_color_var.get(), relief=tk.SUNKEN)
        self.axis_y_color_preview.pack(side=tk.LEFT, padx=4)
        ttk.Button(y_axis_row, text="Выбрать", command=lambda: self.choose_axis_color("y")).pack(side=tk.LEFT)

        # --- Curve management
        curve_frame = ttk.LabelFrame(control_frame, text="Графики")
        curve_frame.grid(row=6, column=0, columnspan=4, sticky="we", pady=(10, 0))

        ttk.Label(curve_frame, text="Подпись").grid(row=0, column=0, sticky="w")
        ttk.Entry(curve_frame, textvariable=self.label_var, width=30).grid(row=0, column=1, columnspan=3, sticky="we")

        button_row = ttk.Frame(curve_frame)
        button_row.grid(row=1, column=0, columnspan=4, pady=6)
        ttk.Button(button_row, text="Добавить", command=self.add_curve).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_row, text="Обновить", command=self.update_selected_curve).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_row, text="Удалить", command=self.remove_selected_curve).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_row, text="Очистить", command=self.clear_curves).pack(side=tk.LEFT, padx=2)

        self.curve_list = tk.Listbox(curve_frame, height=8)
        self.curve_list.grid(row=2, column=0, columnspan=4, sticky="nsew")
        curve_frame.rowconfigure(2, weight=1)
        self.curve_list.bind("<<ListboxSelect>>", self.on_curve_select)

        save_btn = ttk.Button(control_frame, text="Сохранить картинку", command=self.save_figure)
        save_btn.grid(row=7, column=0, columnspan=4, sticky="we", pady=(10, 0))

        # Expand columns where appropriate for nicer layouts
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        self.plot_container = plot_container

    def _build_plot(self) -> None:
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_edit_shortcuts(self, widget: tk.Widget) -> None:
        """Enable standard clipboard and undo/redo shortcuts for the widget."""

        bindings = {
            "c": "<<Copy>>",
            "x": "<<Cut>>",
            "v": "<<Paste>>",
            "a": "<<SelectAll>>",
            "z": "<<Undo>>",
            "y": "<<Redo>>",
        }

        for key, virtual_event in bindings.items():
            for modifier in ("Control", "Command"):
                sequence = f"<{modifier}-{key}>"
                widget.bind(sequence, lambda event, ev=virtual_event: widget.event_generate(ev))

        # Support redo via Shift+Z as commonly used on macOS applications.
        for modifier in ("Control", "Command"):
            sequence = f"<{modifier}-Shift-Z>"
            widget.bind(sequence, lambda event: widget.event_generate("<<Redo>>"))

    def _get_expression(self) -> str:
        """Return the current expression text from the input widget."""

        return self.expr_entry.get("1.0", "end-1c")

    def _set_expression(self, value: str) -> None:
        """Update the expression input with *value* and reset undo history."""

        self.expr_entry.delete("1.0", "end")
        self.expr_entry.insert("1.0", value)
        # Reset undo stack so programmatic changes do not pollute user history.
        self.expr_entry.edit_reset()
        self.expr_entry.edit_modified(False)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------
    def detect_parameters(self, expression: str) -> Sequence[str]:
        """Return sorted parameter names detected in the expression."""

        if not expression.strip():
            return []

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            messagebox.showerror("Ошибка", f"Синтаксическая ошибка в выражении: {exc}")
            return []

        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        parameters = sorted(name for name in names if name not in NUMEXPR_ALLOWED_NAMES)
        return parameters

    def update_parameters(self, preset_values: Optional[Dict[str, float]] = None) -> None:
        """Regenerate parameter fields based on the current expression."""

        expression = self._get_expression()
        parameters = self.detect_parameters(expression)

        # Destroy old widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_vars.clear()
        self.param_frames.clear()

        for index, name in enumerate(parameters):
            ttk.Label(self.param_frame, text=name).grid(row=index, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=(preset_values or {}).get(name, 1.0))
            entry = ttk.Entry(self.param_frame, textvariable=var, width=10)
            entry.grid(row=index, column=1, padx=(4, 0), sticky="we")

            self.param_vars[name] = var
            self.param_frames[name] = entry

    def reset_parameters(self) -> None:
        """Reset all parameter values to the default of 1.0."""

        for var in self.param_vars.values():
            var.set(1.0)

    # ------------------------------------------------------------------
    # Curve management
    # ------------------------------------------------------------------
    def _gather_parameters(self) -> Dict[str, float]:
        params = {}
        for name, var in self.param_vars.items():
            try:
                params[name] = float(var.get())
            except (TypeError, tk.TclError):
                messagebox.showerror("Ошибка", f"Недопустимое значение для параметра '{name}'")
                raise
        return params

    def _create_curve_from_inputs(self) -> Optional[CurveDefinition]:
        expression = self._get_expression().strip()
        if not expression:
            messagebox.showwarning("Предупреждение", "Введите формулу для построения графика")
            return None

        try:
            parameters = self._gather_parameters()
            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            samples = int(self.samples_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Ошибка", "Проверьте корректность числовых значений")
            return None

        label = self.label_var.get().strip() or expression
        color = self.color_var.get()

        curve = CurveDefinition(
            label=label,
            expression=expression,
            parameters=parameters,
            color=color,
            x_min=xmin,
            x_max=xmax,
            samples=max(samples, 10),
        )

        try:
            curve.evaluate()
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return None

        return curve

    def add_curve(self) -> None:
        curve = self._create_curve_from_inputs()
        if curve is None:
            return

        self.curves.append(curve)
        self.curve_list.insert(tk.END, curve.label)
        self.replot_all()

    def update_selected_curve(self) -> None:
        selection = self.curve_list.curselection()
        if not selection:
            messagebox.showinfo("Информация", "Выберите график для обновления")
            return

        index = selection[0]
        curve = self._create_curve_from_inputs()
        if curve is None:
            return

        self.curves[index] = curve
        self.curve_list.delete(index)
        self.curve_list.insert(index, curve.label)
        self.replot_all()

    def remove_selected_curve(self) -> None:
        selection = self.curve_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.curve_list.delete(index)
        del self.curves[index]
        self.replot_all()

    def clear_curves(self) -> None:
        self.curve_list.delete(0, tk.END)
        self.curves.clear()
        self.replot_all()

    def on_curve_select(self, _event: tk.Event) -> None:  # type: ignore[override]
        selection = self.curve_list.curselection()
        if not selection:
            return
        curve = self.curves[selection[0]]
        self._set_expression(curve.expression)
        self.label_var.set(curve.label)
        self.xmin_var.set(curve.x_min)
        self.xmax_var.set(curve.x_max)
        self.samples_var.set(curve.samples)
        self.color_var.set(curve.color)
        self.color_preview.configure(bg=curve.color)
        self.update_parameters(curve.parameters)

    # ------------------------------------------------------------------
    # Plotting and display
    # ------------------------------------------------------------------
    def replot_all(self) -> None:
        self.axes.clear()

        x_axis_color = self.axis_x_color_var.get()
        y_axis_color = self.axis_y_color_var.get()

        if self.logx_var.get():
            self.axes.set_xscale("log")
        else:
            self.axes.set_xscale("linear")

        if self.logy_var.get():
            self.axes.set_yscale("log")
        else:
            self.axes.set_yscale("linear")

        for curve in self.curves:
            try:
                y_values = curve.evaluate()
            except ValueError as exc:
                messagebox.showerror("Ошибка", f"{curve.label}: {exc}")
                continue

            x_values = np.linspace(curve.x_min, curve.x_max, curve.samples)

            if self.logx_var.get() and np.any(x_values <= 0):
                messagebox.showwarning(
                    "Предупреждение",
                    f"График '{curve.label}' пропущен: логарифмическая ось X требует положительных значений",
                )
                continue

            if self.logy_var.get() and np.any(y_values <= 0):
                messagebox.showwarning(
                    "Предупреждение",
                    f"График '{curve.label}' содержит неположительные значения Y, не может быть отображен в логарифмической шкале",
                )
                continue

            self.axes.plot(x_values, y_values, label=curve.label, color=curve.color)

        if self.grid_var.get():
            self.axes.grid(True, which="both", linestyle="--", alpha=0.5)
        else:
            self.axes.grid(False)

        if self.curves:
            self.axes.legend()

        self.axes.set_xlabel("x", color=x_axis_color)
        self.axes.set_ylabel("f(x)", color=y_axis_color)

        self.axes.tick_params(axis="x", colors=x_axis_color)
        self.axes.tick_params(axis="y", colors=y_axis_color)

        for spine_name in ("bottom", "top"):
            if spine_name in self.axes.spines:
                self.axes.spines[spine_name].set_color(x_axis_color)
        for spine_name in ("left", "right"):
            if spine_name in self.axes.spines:
                self.axes.spines[spine_name].set_color(y_axis_color)

        if not self.logy_var.get():
            self.axes.axhline(0, color=x_axis_color, linewidth=1.2, alpha=0.8, label="_nolegend_")
        if not self.logx_var.get():
            self.axes.axvline(0, color=y_axis_color, linewidth=1.2, alpha=0.8, label="_nolegend_")

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Cursor handling
    # ------------------------------------------------------------------
    def _connect_cursor_events(self) -> None:
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self._on_mouse_leave)

        self.cursor_vline = self.axes.axvline(color="gray", lw=0.8, alpha=0.5, visible=False)
        self.cursor_hline = self.axes.axhline(color="gray", lw=0.8, alpha=0.5, visible=False)
        self.cursor_text = self.axes.text(
            0.02,
            0.98,
            "",
            transform=self.axes.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            visible=False,
        )

    def _on_mouse_move(self, event) -> None:
        if not self.cursor_var.get() or event.inaxes != self.axes:
            return

        if event.xdata is None or event.ydata is None:
            return

        self.cursor_vline.set_xdata(event.xdata)
        self.cursor_hline.set_ydata(event.ydata)
        self.cursor_vline.set_visible(True)
        self.cursor_hline.set_visible(True)

        self.cursor_text.set_text(f"x = {event.xdata:.4g}\ny = {event.ydata:.4g}")
        self.cursor_text.set_visible(True)
        self.canvas.draw_idle()

    def _on_mouse_leave(self, _event) -> None:
        if self.cursor_vline is None or self.cursor_hline is None or self.cursor_text is None:
            return
        self.cursor_vline.set_visible(False)
        self.cursor_hline.set_visible(False)
        self.cursor_text.set_visible(False)
        self.canvas.draw_idle()

    def _toggle_cursor(self) -> None:
        if not self.cursor_var.get():
            self._on_mouse_leave(None)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def choose_color(self) -> None:
        color = colorchooser.askcolor(initialcolor=self.color_var.get(), title="Выбор цвета графика")
        if color and color[1]:
            self.color_var.set(color[1])
            self.color_preview.configure(bg=color[1])
            self._apply_color_to_selected_curve()

    def choose_axis_color(self, axis: str) -> None:
        if axis == "x":
            var = self.axis_x_color_var
            preview = self.axis_x_color_preview
            title = "Выбор цвета оси X"
        else:
            var = self.axis_y_color_var
            preview = self.axis_y_color_preview
            title = "Выбор цвета оси Y"

        color = colorchooser.askcolor(initialcolor=var.get(), title=title)
        if color and color[1]:
            var.set(color[1])
            preview.configure(bg=color[1])
            self.replot_all()

    def _apply_color_to_selected_curve(self) -> None:
        selection = self.curve_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.curves[index].color = self.color_var.get()
        self.replot_all()

    def save_figure(self) -> None:
        if not self.curves:
            messagebox.showinfo("Информация", "Нет графиков для сохранения")
            return
        filetypes = [("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("Все файлы", "*.*")]
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
        if filename:
            self.figure.savefig(filename, dpi=300)
            messagebox.showinfo("Успех", f"Изображение сохранено: {filename}")


def main() -> None:
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
