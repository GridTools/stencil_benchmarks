import jinja2


def render(template_file, **kwargs):
    with open(template_file) as file:
        contents = file.read()
    return jinja2.Template(contents).render(**kwargs)
