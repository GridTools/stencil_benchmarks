import os

import jinja2


def render(template_file, **kwargs):
    template_path, template_file = os.path.split(template_file)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = env.get_template(template_file)
    return template.render(**kwargs)
