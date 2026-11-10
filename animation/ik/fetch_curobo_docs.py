"""Mirror the parts of the cuRobo V2 docs an offline h.ik script needs.

The run sandbox has no network (network.allowedDomains = ["localhost"]), so the
agent cannot open nvlabs.github.io. These are the pages it would have wanted.
"""
import re, subprocess, sys, urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://nvlabs.github.io/curobo/latest/"
OUT = Path(sys.argv[1])
# Chosen for what an h.ik script needs, not for completeness. The narrative
# pages first — the API facade pages document the public re-exports, and for
# InverseKinematics that is only "alias of IKSolver".
PAGES = [
    # how to use it
    ("getting-started/inverse_kinematics.html", "guide_inverse_kinematics"),
    ("getting-started/motion_planning.html",    "guide_motion_planning"),
    ("getting-started/forward_kinematics.html", "guide_forward_kinematics"),
    ("getting-started/build_robot_model.html",  "guide_build_robot_model"),
    ("reference/self_collision.html",           "ref_self_collision"),
    ("reference/sphere_fitting.html",           "ref_sphere_fitting"),
    ("reference/runtime_configuration.html",    "ref_runtime_configuration"),
    ("guides/optimization_motion.html",         "guide_optimization_motion"),
    # the API surface
    ("api/curobo.motion_planner.html",          "api_motion_planner"),
    ("api/curobo.types.html",                   "api_types"),
    ("api/curobo.kinematics.html",              "api_kinematics"),
    ("api/curobo.robot_builder.html",           "api_robot_builder"),
    ("api/curobo.batch_motion_planner.html",    "api_batch_motion_planner"),
    ("api/curobo.sphere_fit.html",              "api_sphere_fit"),
    ("api/curobo.scene.html",                   "api_scene"),
    ("api/curobo.inverse_kinematics.html",      "api_inverse_kinematics"),
]

class Article(HTMLParser):
    """Slice out <article role="main">…</article> from a Furo page."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.depth = 0; self.parts = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if self.depth == 0 and tag == "article" and d.get("role") == "main":
            self.depth = 1; return
        if self.depth:
            if tag == "article": self.depth += 1
            self.parts.append(self.get_starttag_text())
    def handle_startendtag(self, tag, attrs):
        if self.depth: self.parts.append(self.get_starttag_text())
    def handle_endtag(self, tag):
        if not self.depth: return
        if tag == "article":
            self.depth -= 1
            if self.depth == 0: return
        self.parts.append(f"</{tag}>")
    def handle_data(self, d):
        if self.depth: self.parts.append(d)
    def handle_entityref(self, n):
        if self.depth: self.parts.append(f"&{n};")
    def handle_charref(self, n):
        if self.depth: self.parts.append(f"&#{n};")

def _clean(md: str) -> str:
    """Strip the Furo/pandoc scaffolding that survives the HTML conversion.

    Sphinx wraps every section in nested <div>s and every heading in a
    permalink <a>; pandoc passes those through as raw HTML. Left in, they are
    roughly half the bytes and all of the noise.
    """
    md = re.sub(r'<a href="#[^"]*" class="headerlink"[^>]*>[^<]*</a>', "", md)
    md = re.sub(r'\[\]\(#[^)]*"Link to this[^"]*"\)', "", md)
    md = re.sub(r"</?div[^>]*>", "", md)
    md = re.sub(r"</?span[^>]*>", "", md)
    md = re.sub(r"<!-- -->", "", md)
    md = re.sub(r"[ \t]+$", "", md, flags=re.M)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


OUT.mkdir(parents=True, exist_ok=True)
for path, name in PAGES:
    url = BASE + path
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    a = Article(); a.feed(html)
    body = "".join(a.parts)
    md = subprocess.run(["pandoc", "-f", "html", "-t", "gfm", "--wrap=none"],
                        input=body, capture_output=True, text=True).stdout
    md = _clean(md)
    header = (f"<!-- Mirrored from {url}\n"
              f"     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.\n"
              f"     Local copy because a dycraft run has no network.\n"
              f"     Refresh: python animation/ik/fetch_curobo_docs.py"
              f" animation/ik/curobo_docs -->\n\n")
    (OUT / f"{name}.md").write_text(header + md + "\n")
    print(f"  {name+'.md':28s} {len(md)//1024:4d} KB")
