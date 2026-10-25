# SPDX-License-Identifier: GPL-2.0
import json

from support import TempDirTestCase, commit_file, git, load

prepare = load("prepare-recipe")

OLD = "1" * 40
PKGBUILD = f"""\
_auroraver=7.2.10
_aurorarel=1
pkgrel=2

_commit={OLD}

pkgver="${{_auroraver}}.aurora${{_aurorarel}}"
source=(validate-config.py configure.sh "linux::git+https://github.com/omacom/linux-aurora.git#commit=${{_commit}}"
        rust-toolchain.toml bindgen-kernel.py config)
sha256sums=('x' 'y' 'SKIP' 'z' 'w' 'v')
"""


class RewriteTest(TempDirTestCase):
    def test_changes_only_the_three_lines(self):
        commit = "c" * 40
        text, record = prepare.rewrite(PKGBUILD, commit, 7)
        self.assertIn(f"_commit={commit}\n", text)
        self.assertIn("pkgrel=2.7\n", text)
        self.assertIn('"linux::git+file://${startdir}/aurora-ci-source#commit=${_commit}"', text)
        self.assertNotIn("github.com/omacom/linux-aurora.git", text)
        changed = [(a, b) for a, b in zip(PKGBUILD.splitlines(), text.splitlines()) if a != b]
        self.assertEqual(len(changed), 3)
        self.assertEqual(record["recipe_commit"], OLD)
        self.assertEqual(record["pkgrel"], "2.7")

    def test_refuses_an_unexpected_recipe(self):
        cases = {
            "no commit pin": PKGBUILD.replace(f"_commit={OLD}\n", ""),
            "two commit pins": PKGBUILD + f"_commit={OLD}\n",
            "short commit pin": PKGBUILD.replace(OLD, "1234567"),
            "pkgrel with a subrelease": PKGBUILD.replace("pkgrel=2\n", "pkgrel=2.1\n"),
            "no pkgrel": PKGBUILD.replace("pkgrel=2\n", ""),
            "a different source": PKGBUILD.replace("omacom/linux-aurora.git", "elsewhere/linux.git"),
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(prepare.RecipeError):
                prepare.rewrite(text, "c" * 40, 1)

    def test_refuses_bad_arguments(self):
        for commit, revision in (("abc", 1), ("C" * 40, 1), ("c" * 40, 0), ("c" * 40, "x")):
            with self.subTest(commit=commit, revision=revision), self.assertRaises(prepare.RecipeError):
                prepare.rewrite(PKGBUILD, commit, revision)


class SourceTest(TempDirTestCase):
    def make_kernel(self):
        kernel = self.init_repo("kernel")
        commit_file(kernel, "Makefile", "VERSION = 7\n", "first")
        return kernel, commit_file(kernel, "README", "second\n", "second")

    def test_main_builds_a_bare_source_holding_the_commit(self):
        kernel, head = self.make_kernel()
        recipe = self.tmp / "linux-aurora"
        recipe.mkdir()
        (recipe / "PKGBUILD").write_text(PKGBUILD)
        record = self.tmp / "record.json"
        _, summary = self.github_env()
        status = prepare.main(["--recipe", str(recipe), "--kernel", str(kernel), "--commit", head,
                               "--revision", "12", "--record", str(record)])
        self.assertEqual(status, 0)
        source = recipe / prepare.SOURCE_DIR
        self.assertEqual(git(source, "rev-parse", "HEAD"), head)
        self.assertEqual(git(source, "rev-parse", "--is-bare-repository"), "true")
        self.assertTrue((source / "shallow").is_file())
        self.assertIn("pkgrel=2.12", (recipe / "PKGBUILD").read_text())
        self.assertEqual(json.loads(record.read_text())["commit"], head)
        self.assertIn("Recipe override", summary.read_text())

    def test_main_leaves_the_recipe_alone_on_error(self):
        kernel, _ = self.make_kernel()
        recipe = self.tmp / "linux-aurora"
        recipe.mkdir()
        (recipe / "PKGBUILD").write_text(PKGBUILD)
        self.github_env()
        status = prepare.main(["--recipe", str(recipe), "--kernel", str(kernel), "--commit", "d" * 40,
                               "--revision", "1"])
        self.assertEqual(status, 1)
        self.assertEqual((recipe / "PKGBUILD").read_text(), PKGBUILD)

    def test_refuses_to_reuse_a_source_directory(self):
        kernel, head = self.make_kernel()
        recipe = self.tmp / "linux-aurora"
        (recipe / prepare.SOURCE_DIR).mkdir(parents=True)
        with self.assertRaises(prepare.RecipeError):
            prepare.make_source(recipe, kernel, head)
