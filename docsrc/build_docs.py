import json
import os
from os import path

import click
import networkx as nx

MODULE_NAME = 'dair_pll'
DOCS_DIR = path.dirname(__file__)
PROJECT_DIR = path.dirname(DOCS_DIR)
SOURCE_DIR = path.join(DOCS_DIR, 'source')
BUILD_DIR = path.join(DOCS_DIR, 'docs')
PUBLISH_DIR = path.join(PROJECT_DIR, 'docs')
MODULE_DIR = path.join(PROJECT_DIR, MODULE_NAME)
TEMPLATE_DIR = path.join(DOCS_DIR, 'templates')
INDEX_RST = path.join(SOURCE_DIR, 'index.rst')
MODULES_RST = path.join(SOURCE_DIR, 'modules.rst')
MODULE_RST = path.join(SOURCE_DIR, f'{MODULE_NAME}.rst')
INDEX_TEXT_RST = path.join(SOURCE_DIR, 'index_text.rst.template')
DEP_JSON_FILE = path.join(DOCS_DIR, 'graph.json')


# regenerate .rst's
def build(regenerate_deps: bool = False):
    # remove any old documentation
    os.system(f'rm {SOURCE_DIR}/{MODULE_NAME}.*')

    # generate new .rst's
    os.system(
        f'sphinx-apidoc -f -o {SOURCE_DIR} {MODULE_DIR} -e --templatedir'
        f'={TEMPLATE_DIR}')

    # generate index.rst
    with open(INDEX_TEXT_RST, 'r', encoding='utf-8') as index_text_file:
        index_text = index_text_file.readlines()

    toc_setup = [
        '',
        '.. toctree::',
        '   :maxdepth: 8',
        '   :caption: Submodules:',
        '   :titlesonly:',
        '',
    ]
    index_text.extend([f'{line}\n' for line in toc_setup])

    # build dependency graph
    if regenerate_deps:
        os.system(
            f'pydeps --only {MODULE_NAME} --show-deps {MODULE_DIR} --no-output'
            ' --debug'
            f' > {DEP_JSON_FILE}')

    with open(DEP_JSON_FILE, 'r', encoding='utf-8') as dep_file:
        dep_graph = json.load(dep_file)
    # print(dep_graph)
    graph = nx.DiGraph()
    edges = []
    excludes = ['dair_pll']
    main_document = 'dair_pll.drake_experiment'
    for module, module_details in dep_graph.items():
        if 'imports' not in module_details:
            continue
        if module in excludes:
            continue
        for imported_module in module_details['imports']:
            if imported_module in excludes:
                continue
            edges.append((module, imported_module))
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('import cycle detected in package!')

    mod_list = []

    # define tiebreak ordering for topological dependency sort
    def tiebreak(x):
        # Put full module first.
        if not '.' in x:
            return 0

        # Next, put utility files.
        if '_utils' in x:
            return -1

        # Last, everything else.
        return -2

    # sort submodules by top-down dependency
    # pdb.set_trace()
    for module in nx.lexicographical_topological_sort(graph, key=tiebreak):
        mod_list.append(str(module))

    # Reverse to bottom-up dependency
    mod_list.reverse()

    # sort for nodes relevant to docs
    reachable = nx.single_source_shortest_path(graph, main_document).keys()
    not_reached = [module for module in mod_list if not module in reachable]
    mod_list = [module for module in mod_list if module in reachable]

    # add modules to table of contents.
    for module in mod_list:
        index_text.append(f'   {module}\n')

    index_text.append(f'   bibliography\n')

    with open(INDEX_RST, 'w', encoding='utf-8') as index_file:
        index_file.write(''.join(index_text))

    # remove unused files
    remove_files = [path.join(SOURCE_DIR, f'{file_to_remove}.rst')
                    for file_to_remove in set(not_reached + excludes)]
    remove_files += [MODULES_RST]

    for filename in remove_files:
        os.system(f'rm {filename}')

    # build html
    os.system(f'sphinx-build -b html {SOURCE_DIR} {PUBLISH_DIR}')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--regenerate_deps/--use_stored_deps', default=False,
              help="reorder submodules by dependency")
def build_command(regenerate_deps: bool):
    build(regenerate_deps)


if __name__ == '__main__':
    build_command()
