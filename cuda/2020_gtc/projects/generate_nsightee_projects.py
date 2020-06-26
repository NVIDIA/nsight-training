#!/usr/bin/python

import argparse
import os
import shutil
import sys

skipped_steps = [2]

class SourceFile:
    def __init__(self, src, dst_name=None):
        self.src = src
        self.dst_name = dst_name

    def copy(self, dst, step):
        if not os.path.isfile(self.src):
            print("Failed to find src file " + self.src)
            return False

        if self.dst_name == None:
            dst_file = os.path.basename(self.src)
        else:
            dst_file = self.dst_name.replace("VAR(STEP)", str(step))

        dst_loc = dst + "/" + dst_file
        print("Copying " + self.src + " to " + dst_loc)
        shutil.copyfile(self.src, dst_loc)
        return True

class TemplateFile:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.template = []

    def load(self):
        if not os.path.isfile(self.src):
            print("Failed to find template file " + self.src)
            return False

        with open(self.src) as f:
            for line in f:
                self.template.append(line)

        return True

    def write(self, path, step, platform, src_dir):
        with open(path + "/" + self.dst, "w") as f:
            for line in self.template:
                tmp = line
                tmp = tmp.replace("VAR(STEP)", step)
                tmp = tmp.replace("VAR(PLATFORM)", platform)
                tmp = tmp.replace("VAR(DST_DIR)", path)
                tmp = tmp.replace("VAR(SRC_DIR)", "SRC_DIR=" + src_dir)
                f.write(tmp)


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Lab content directory")
    parser.add_argument("--dst", help="Project destination directory")
    parser.add_argument("--steps", help="Number of steps to generate, starting with 0")
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    return parser.parse_args()

def main():
    args = parse_command_line()
    if args.src == None:
        args.src = "../code"

    if args.steps == None:
        args.steps = 7

    if not args.dst:
        print("Missing --dst flag")
        return 1

    # create projects in workspace
    src_files = [
        SourceFile(os.path.abspath(args.src + "/imgserver.cu"), "imgserver_VAR(STEP).cu"),
        SourceFile(os.path.abspath(args.src + "/Makefile")),
        SourceFile(os.path.abspath(args.src + "/findgllib.mk")),
    ]

    templates = [TemplateFile("project.template", ".project"), TemplateFile("cproject.template", ".cproject")]
    for template in templates:
        if not template.load():
            print("Failed to load " + template.src)
            return 1

    platform = "desktop=1"
    arch = "x86_64"
    compute_proj_ext = "nsight-cuproj"

    for step in range(0, int(args.steps) + 1):
        if step in skipped_steps:
            print("Skipping step " + str(step))
            continue

        dst_dir = args.dst + "/step_" + str(step)
        print("Creating " + dst_dir)

        if os.path.exists(dst_dir):
            if not args.force:
                print("Target directory " + dst_dir + " exists, use --force to overwrite")
                return 1

            shutil.rmtree(dst_dir)

        os.mkdir(dst_dir)

        for src_file in src_files:
            src_file.copy(dst_dir, step)

        for template in templates:
            template.write(dst_dir, str(step), platform, os.path.abspath(args.src) + "/")

        # create launch config for this project
        print("Creating launch config")
        launch_config = TemplateFile("launch_{0}.launch".format(arch), "launch_step_{0}.launch".format(step))
        if not launch_config.load():
            print("Failed to load " + launch_config)
            return 1

        launch_config.write(args.dst, str(step), platform, os.path.abspath(args.src) + "/")


    # copy nsight systems/compute projects to workspace
    perf_proj_templates = [TemplateFile("nsight_compute_" + arch + "." + compute_proj_ext, "nsight_compute." + compute_proj_ext)]
    for template in perf_proj_templates:
        if not template.load():
            print("Failed to load " + template.src)
            return 1

        template.write(args.dst, str(step), platform, os.path.abspath(args.src) + "/")


    # copy image resources to workspace
    img_dir = args.dst + "/img"
    print("Copying img dir to " + img_dir)
    if os.path.exists(img_dir):
        if not args.force:
            print("Target directory " + img_dir + " exists, use --force to overwrite")
            return 1

        shutil.rmtree(img_dir)

    shutil.copytree(os.path.abspath(args.src + "/../img"), img_dir)

    print("done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
