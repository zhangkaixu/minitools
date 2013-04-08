#!/usr/bin/python3
import sys
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',type=str, help='')
    parser.add_argument('--output',type=str, help='')
    parser.add_argument('--before',type=str,nargs='*', help='')
    parser.add_argument(dest='mid',type=str,nargs='*', help='')
    parser.add_argument('--after',type=str,nargs='*', help='')
    parser.add_argument('--if',dest='iiff',type=str, help='')
    parser.add_argument('--with_weight',action="store_true", help='')
    args = parser.parse_args()

    instream=open(args.input) if args.input else sys.stdin
    outstream=open(args.output,'w') if args.output else sys.stdout

    if args.before :
        for c in args.before :
            exec(c)
    for line in sys.stdin :
        line=line.strip()
        if args.iiff:
            if eval(args.iiff) :
                print(line)
        if args.mid :
            for c in args.mid :
                exec(c)
    if args.after :
        for c in args.after :
            exec(c)
