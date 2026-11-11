#!/usr/bin/env python3
"""Write a variant of a Strata run config: variant.py <base.json> <out.json> [key=JSON ...] [-arg FLAG] [+arg FLAG VALUE...]

  key=JSON         set a top-level field, e.g. gpu=[1]  gpu_roles=null (null removes the key)
  -arg FLAG        remove FLAG (and its value, if the next item is not a flag) from "args"
  +arg FLAG [V]    append FLAG (and V) to "args" (an existing FLAG is removed first)
  env.NAME=VALUE   set an environment variable in "env"
"""
import json
import sys


def drop(args, flag):
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 1
            if i < len(args) and not args[i].startswith("--"):
                i += 1
            continue
        out.append(args[i])
        i += 1
    return out


def main(argv):
    base, out, rest = argv[1], argv[2], argv[3:]
    cfg = json.load(open(base))
    args = list(cfg.get("args", []))
    i = 0
    while i < len(rest):
        item = rest[i]
        if item == "-arg":
            args = drop(args, rest[i + 1])
            i += 2
        elif item == "+arg":
            flag = rest[i + 1]
            args = drop(args, flag) + [flag]
            i += 2
            if i < len(rest) and not rest[i].startswith(("+arg", "-arg")) and "=" not in rest[i]:
                args.append(rest[i])
                i += 1
        elif item.startswith("env."):
            k, v = item[4:].split("=", 1)
            cfg.setdefault("env", {})[k] = v
            i += 1
        else:
            k, v = item.split("=", 1)
            val = json.loads(v)
            if val is None:
                cfg.pop(k, None)
            else:
                cfg[k] = val
            i += 1
    cfg["args"] = args
    json.dump(cfg, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv)
