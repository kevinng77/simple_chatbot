


def get_link_map(file_path):
    """
    Returns:
        dictionary: mapping entity name to its official name in the database.
    """
    with open(file_path,"r") as fp:
        lines = fp.readlines()
    mapping = {}
    for line in lines:
        words = line.strip().split(' ')
        if len(words) == 1:  # only official name is recorded
            continue
        else:
            official_name = words[0]
            for word in words[1:]:
                mapping[word] = official_name
    return mapping


def link_entity(entity, mapping):
    """
    given entity name, find the official entity name in the database.
    Args:
        entity (str): name of entity
    Returns:
        str: official name of the given entity
    """
    result = mapping.get(entity,None)
    return result


if __name__ == '__main__':
    mapping = get_link_map(file_path="data/place.txt")
    entity = "国家体育场"
    result = link_entity(entity, mapping)
    print(result)