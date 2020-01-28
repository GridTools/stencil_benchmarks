import os
from typing import Any, Dict

import jinja2


def render(template_file: str, **kwargs: Dict[str, Any]) -> str:
    template_path, template_file = os.path.split(template_file)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = env.get_template(template_file)
    return template.render(**kwargs)
