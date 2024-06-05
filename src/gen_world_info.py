import sys
import json
import base64
import collections

from deepspeed.launcher.runner import fetch_hostfile
from deepspeed.launcher.runner import parse_inclusion_exclusion
from deepspeed.launcher.runner import encode_world_info


def main():
    num_nodes = int(sys.argv[1])
    num_gpus = int(sys.argv[2])
    host_file = sys.argv[3]

    resource_pool = fetch_hostfile(host_file)
    active_resources = parse_inclusion_exclusion(resource_pool, "", "")

    if num_nodes > 0:
        updated_active_resources = collections.OrderedDict()
        for count, hostname in enumerate(active_resources.keys()):
            if num_nodes == count:
                break
            updated_active_resources[hostname] = active_resources[hostname]
        active_resources = updated_active_resources

    if num_gpus > 0:
        updated_active_resources = collections.OrderedDict()
        for hostname in active_resources.keys():
            updated_active_resources[hostname] = list(range(num_gpus))
        active_resources = updated_active_resources

    world_info_base64 = encode_world_info(active_resources)
    print(world_info_base64)

    # 反向解码验证
    # world_info = base64.urlsafe_b64decode(world_info_base64)
    # world_info = json.loads(world_info)
    # print(f"decode world_info_base64 = {world_info}")


if __name__ == '__main__':
    main()
