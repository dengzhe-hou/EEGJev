"""Recalculate the source-balanced results and redraw Figure 3a from 13 source means."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MID, FAINT, GRID = "#1F1F1F", "#5F5F5F", "#9A9A9A", "#DFE1E0"
JEV = "#1C9A9A"


plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                      "font.size":7, "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
                      "lines.solid_capstyle":"round", "figure.facecolor":"white", "savefig.facecolor":"white"})

def panel(f,box,letter,title,subtitle="",*,track_color=None):
    x,y,w,h=box; H=f.get_figheight()*25.4
    ax=f.add_axes([x/180,(H-y-h)/H,w/180,h/H],label=letter)
    ax.set(xlim=(0,w),ylim=(h,0));ax.set_axis_off()
    txt(ax,0,2.5,letter,bold=True,size=8.5)
    txt(ax,4.8,2.5,title,bold=True,size=7.6)
    if subtitle: txt(ax,4.8,7.1,subtitle,size=6.3,col=MID)
    if track_color is not None:
        # Locate the colour key beside the subtitle and keep the small text in dark ink.
        ln(ax,0.1,7.1,2.85,7.1,col=track_color,lw=2.0,z=4)
    return ax

def txt(ax,x,y,s,*,col=INK,ha="left",va="center",bold=False,size=7,rot=0):
    return ax.text(x,y,s,color=col,ha=ha,va=va,fontweight="bold" if bold else "normal",fontsize=max(size,6.2),
                   rotation=rot,rotation_mode="anchor",linespacing=1.24,clip_on=False,zorder=8)

def ln(ax,x1,y1,x2,y2,*,col=GRID,lw=.55,ls="-",alpha=1,z=1):
    ax.plot([x1,x2],[y1,y2],color=col,lw=lw,ls=ls,alpha=alpha,clip_on=False,zorder=z)

def pt(ax,x,y,*,col=INK,mark="o",sz=3.7,hollow=False,alpha=1,z=5):
    ax.plot(x,y,marker=mark,markersize=sz,linestyle="none",markerfacecolor="white" if hollow else col,
            markeredgecolor=col,markeredgewidth=.8,color=col,alpha=alpha,clip_on=False,zorder=z)

def scale(lo,hi,u,v): return lambda z: u + (z-lo)*(v-u)/(hi-lo)

def tag(ax,x,y,label,**kw):
    pt(ax,x,y,**kw); txt(ax,x+2.6,y,label,size=6.3,col=MID)

def xaxis(ax,s,ticks,extent,y,*,zero=None,dec=0):
    for v in ticks:
        x=s(v)
        if zero is not None and v==zero: ln(ax,x,extent,x,y,col=FAINT,lw=.7,ls=(0,(2,2)))
        ln(ax,x,y,x,y+.85,col=MID,lw=.55)
        txt(ax,x,y+3.05,f"{v:.{dec}f}".replace("-","−"),ha="center",size=6.2,col=MID)
    ln(ax,s(ticks[0]),y,s(ticks[-1]),y,col=MID,lw=.55)

from matplotlib import font_manager

for face in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    font_manager.fontManager.addfont(Path(__file__).with_name("fonts") / face)
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

def fixed_options_panel(figure, sources):
    """Two aligned source-level plots. Uniform choice applies to attribute selection only."""
    ax = panel(
        figure, (3, 2, 174, 74), "a", "Does the answer follow the question?",
        "Same EEG and options, two questions about familiar attributes", track_color=JEV,
    )
    tag(ax, 111, 7.1, "EEGJev", col=JEV, mark="D", sz=3.6)
    tag(ax, 133, 7.1, "Uniform option choice", col=MID, mark="o", hollow=True, sz=3.5)

    metrics = (
        ("Requested attribute (accuracy, %)", "attribute_accuracy_pct", 24, 83, 95),
        ("Specific answer (balanced accuracy, %)", "answer_balanced_accuracy_pct", 101, 160, 173),
    )
    txt(ax, 0, 13.3, "Test source", size=6.9, bold=True)
    for title, _, left, right, _ in metrics:
        txt(ax, left, 13.3, title, size=7.0, bold=True)
        position = scale(0, 100, left, right)
        for tick in (0, 50, 100):
            ln(ax, position(tick), 16.7, position(tick), 65.8, col=GRID, lw=.42)
        xaxis(ax, position, [0, 50, 100], 16.7, 68.1)

    for index, source in enumerate(sources):
        name = source["source"]
        overall = name == "Source mean"
        y = 64.1 if overall else 18.8 + 3.2 * index
        if overall:
            ln(ax, 0, 60.8, 173, 60.8, col=FAINT, lw=.55)
        txt(ax, 0, y, name, size=6.8, bold=overall)
        for _, key, left, right, value_x in metrics:
            position = scale(0, 100, left, right)
            if key == "attribute_accuracy_pct":
                chance = float(source["uniform_attribute_accuracy_pct"])
                pt(ax, position(chance), y, col=MID, sz=3.4, hollow=True)
            value = float(source[key])
            pt(ax, position(value), y, col=JEV, mark="D", sz=4.2 if overall else 3.6)
            label = f"{value:.2f}" if overall else f"{value:.1f}"
            txt(ax, value_x, y, label, size=6.8, bold=overall, ha="right")


def run(output):
    source = Path(__file__).parent.parent / "source-data" / "figure-3a-by-source.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    names = [row["source"] for row in rows]
    if len(set(names)) != 13 or len(names) != 13 or "Source mean" in names:
        raise ValueError("Expected exactly one input row per source and no precomputed overall row")
    means = {name: fmean(float(row[name]) for row in rows) for name in fields if name != "source"}
    panel_rows = [*rows, {"source": "Source mean", **means}]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "recomputed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panel_rows)
    summary = {"source_count": len(rows), "weighting": "equal per source", **means}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure = plt.figure(figsize=(180 / 25.4, 78 / 25.4))
    fixed_options_panel(figure, panel_rows)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(output / f"figure3a.{suffix}", dpi=600)
    plt.close(figure)
    print(f"{len(rows)} equally weighted sources")
    print(f"Requested attribute accuracy   {means['attribute_accuracy_pct']:.2f}%")
    print(f"Uniform option choice          {means['uniform_attribute_accuracy_pct']:.2f}%")
    print(f"Specific answer balanced acc.  {means['answer_balanced_accuracy_pct']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    run(parser.parse_args().output)
