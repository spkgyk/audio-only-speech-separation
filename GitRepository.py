###
# Author: Kai Li
# Date: 2022-06-01 17:20:48
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-06-01 18:14:13
###
import os
from git.repo import Repo
from git.repo.fun import is_git_dir
from rich import print


class GitRepository(object):
    """
    git仓库管理
    """

    def __init__(self, local_path, repo_url, branch="dev"):
        self.local_path = local_path
        self.repo_url = repo_url
        self.repo = None
        self.initial(repo_url, branch)

    def initial(self, repo_url, branch):
        """
        初始化git仓库
        :param repo_url:
        :param branch:
        :return:
        """
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        self.repo = Repo(self.local_path)

    def pull(self):
        """
        从线上拉最新代码
        :return:
        """
        self.repo.git.pull()

    def push(self, COMMIT_MESSAGE):
        """
        推送最新代码
        :return:
        """
        self.repo.git.add("--all")
        self.repo.index.commit(COMMIT_MESSAGE)

        origin = self.repo.remote(name="origin")
        origin.push()
        print('A new commit "{}" is pushed successfully!'.format(COMMIT_MESSAGE))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message",
        default=None,
        help="Commit message",
    )
    args = parser.parse_args()
    local_path = os.getcwd()
    repo = GitRepository(local_path, "https://git.likai.show/Meta2ML/Look2Hear.git")
    print("Pulling the code from github!!")
    repo.pull()
    if args.message != None:
        repo.push(args.message)
