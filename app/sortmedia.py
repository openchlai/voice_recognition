#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import hashlib

from time import time


BASE = os.path.dirname(os.path.realpath(__file__))

def main(dat):
    print("""Sort Initial Media """, dat)

    if not os.path.isdir(BASE + "/media"):
        print("""Create Base Media Directory""")

        cmd = "mkdir -p " + BASE + "/media"
        os.system(cmd)

    if all(key in dat for key in ['path', 'dest']):
        print("""Check for Demo Media""")

        if not os.path.isdir(BASE + "/media/" + dat['path']):
            print("""Create MP3 Directory""")
            cmd = "mkdir -p " + BASE + "/media/" + dat['path']
            os.system(cmd)

        if not os.path.isdir(BASE + "/media/" + dat['dest']):
            print("""Create MP3 Directory""")
            cmd = "mkdir -p " + BASE + "/media/" + dat['dest']
            os.system(cmd)

        if os.path.isdir(BASE + "/media/" + dat['path']) and os.listdir(
            BASE + "/media/" + dat['path']):
            print("""Use Demo Media""")

            media = {}

            for x in os.listdir(BASE + "/media/" + dat['path']):
                print("""Original Media ID """, x)

                endf = x.split(".")

                if len(endf) > 1 and endf[-1] in ['mp3', 'mp4']:
                    # print("""Original Media Encode """)

                    f = hashlib.sha256(x.encode()).hexdigest() + "." + endf[-1]

                    media[f] = {}
                    media[f]['parent'] = x
                    media[f]['models'] = {}
                    media[f]['reference'] = False
                    media[f]['created'] = int(time())

                    print("""Media Encode File """, f)

                    if os.path.isfile(BASE + "/media/" + dat['dest'] + "/" + f):
                        print("""Remove Media File """, f)
                        cmd = "rm " + BASE + "/media/" + dat['dest'] + "/" + f
                        os.system(cmd)
                        media[f]['deleted'] = int(time())

                    cmd = "cp '" + BASE + "/media/" + dat['path'] + "/" + x
                    cmd += "' " + BASE + "/media/" + dat['dest'] + "/" + f

                    # print("""Copy Media File """, cmd)
                    os.system(cmd)

            if len(media):
                print("""Process Master JSON""")

                with open(BASE + "/media/" + dat['dest'] + "/master.json", 'w') as fp:
                    fp.write(json.dumps(media, indent=4))

    return True


if __name__ == '__main__':
    x = main({"path": "demo", "dest": "volume"})
    print(x)
