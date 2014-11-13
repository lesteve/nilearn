import re

import tempita


class Container(object):
    def __init__(self):
        self._children = []

    def add(self, obj):
        self._children.append(obj)
        return self

    def get_children_html(self):
        children_html = (obj.get_html() for obj in self._children)
        content = '\n'.join(children_html)
        return content


class Report(Container):
    def __init__(self):
        super(Report, self).__init__()
        self.template = tempita.Template.from_filename(
            './templates/main_template.html')

    def get_sidebar_html(self):
        sidebar_template = tempita.Template(
            '{{for section in sections}}'
            '  <li>\n'
            '    <a href="#{{section.html_id}}">{{section.title}}</a>\n'
            '  </li>\n'
            '{{endfor}}')
        # sidebar only keeps level 1 sections
        # template could be extended to add level 2
        sections = (obj for obj in self._children
                    if isinstance(obj, Section) and obj.level == 1)
        return sidebar_template.substitute(sections=sections)

    def get_html(self):
        content = self.get_children_html()
        return self.template.substitute(
            content=content,
            sidebar_content=self.get_sidebar_html())

    def save_html(self, filename):
        html = self.get_html()
        with open(filename, 'w') as f:
            f.write(html)


class Section(Container):
    def __init__(self, title, level=1):
        super(Section, self).__init__()
        # Keep only letters, digits and hyphens which is a safe subset
        # of what the 'id' attribute can contain
        html_id = re.sub(r'[^\w-]+', '-', title)

        self.html_id = html_id
        self.level = level
        self.title = title

    def get_html(self):
        content = self.get_children_html()
        template = tempita.Template(
            '<div id="{{html_id}}">\n'
            '  <h{{level}}>{{title}}</h{{level}}>'
            '  {{content}}'
            '</div>')

        return template.substitute(title=self.title,
                                   level=self.level,
                                   html_id=self.html_id,
                                   content=content)


class Paragraph(object):
    def __init__(self, content):
        self.content = content

    def get_html(self):
        template = tempita.Template(
            '<p>\n{{content}}\n</p>')
        return template.substitute(content=self.content)


class Image(object):
    def __init__(self, filename, caption=None):
        self.filename = filename
        self.caption = caption

    def get_html(self):
        template = tempita.Template(
            '<div>\n'
            '  <div class="thumbnail">\n'
            '    <img src={{filename}} alt="Can not find file {{filename}}"/>\n'
            '  </div>\n'
            '  {{if caption}}\n'
            '  <div class="caption">\n'
            '  {{caption}}'
            '  </div>\n'
            '  {{endif}}'
            '</div>')

        return template.substitute(filename=self.filename,
                                   caption=self.caption)


class Table(object):
    def __init__(self, rows, headers=None, html_class='table table-striped'):
        self.rows = rows
        self.headers = headers or []
        self.html_class = html_class

    def get_html(self):
        template = tempita.Template(
            '<table class="{{html_class}}">\n'
            '  <thead>\n'
            '    <tr>\n'
            '      {{for header in headers}}\n'
            '        <th>{{header}}</th>\n'
            '      {{endfor}}\n'
            '    </tr>\n'
            '  </thead>\n'
            '  <tbody>\n'
            '    {{for row in rows}}\n'
            '      <tr>\n'
            '      {{for value in row}}\n'
            '        <td>{{value}}</td>\n'
            '      {{endfor}}\n'
            '      </tr>\n'
            '    {{endfor}}\n'
            '  </tbody>\n'
            '<table>'
        )

        return template.substitute(rows=self.rows,
                                   headers=self.headers,
                                   html_class=self.html_class)
