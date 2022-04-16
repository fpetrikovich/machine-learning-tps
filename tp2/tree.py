def export_tree(tree):
    global node_counter
    node_counter = 0
    dot_file = open("tree.dot", "w")
    dot_file.write("digraph {\nsize = \"10,20!\";\nratio = \"fill\";\nrankdir=\"LR\";overlap=false;\n")
    draw_tree_dot(tree, dot_file)
    dot_file.write("}")
    dot_file.close()

def draw_tree_text(tree, height):
    if type(tree) is dict:
        for i in range(height):
            print("| ", end="")
        print(str(tree['attribute']).upper())
        for child in tree['children'].keys():
            for i in range(height+1):
                print("| ", end="")
            print(child)
            draw_tree_rec(tree['children'][child], height+2)
        return node_name
    else:
        for i in range(height-1):
            print("| ", end="")
        print("  ***",str(tree).upper(),"***")

def draw_tree_dot(tree, dot_file):
    global node_counter
    node_name = "n"+str(node_counter)
    if type(tree) is dict:
        dot_file.write("\t"+node_name +" [ fontsize=30 shape=\"box\" label=\"" +str(tree['attribute']).upper() +"\" ]\n")
        node_counter += 1
        for child in sorted(tree['children'].keys()):
            child_node_name = draw_tree_dot(tree['children'][child], dot_file)
            dot_file.write("\t"+node_name +" -> " +child_node_name +" [ fontsize=20 xlabel=\"" +str(child) +"\" ]\n")
    else:
        if(str(tree) == '0'):
            color_property = "style=filled fillcolor=\"darksalmon\""
        else:
            color_property = "style=filled fillcolor=\"darkolivegreen1\""
        dot_file.write("\t"+node_name +" [ " +color_property +"label=\"" +str(tree).upper() +"\" ]\n")
        node_counter += 1
    return node_name