import { Head } from "fresh/runtime";
import { define } from "../utils.ts";
import GridExplorer from "../islands/GridExplorer.tsx";
import TileMandelbrot from "../islands/TileMandelbrot.tsx";
import LifeStencil from "../islands/LifeStencil.tsx";

export default define.page(function Playground() {
  return (
    <div class="page">
      <Head>
        <title>Playground · tileworld</title>
      </Head>
      <header class="page-header">
        <h1>Playground</h1>
        <p class="lede">
          Every demo runs on your CPU, in the browser. They show how cuTile
          divides work between tile programs, not how fast a GPU does it.
        </p>
      </header>

      <section class="demo" aria-labelledby="grid-demo">
        <div class="demo-intro">
          <h2 id="grid-demo">From partition to launch grid</h2>
          <p>
            Choose a tensor shape and a partition. Each block is one tile
            program that owns a separate piece of the output. Hover or tap a
            block to see its <code>program_id</code>{" "}
            and the region it writes. The{" "}
            <a href="/wiki/partitioning-and-the-grid">partitioning note</a>{" "}
            explains the rules.
          </p>
        </div>
        <GridExplorer />
      </section>

      <section class="demo" aria-labelledby="mandel-demo">
        <div class="demo-intro">
          <h2 id="mandel-demo">A Mandelbrot, one tile at a time</h2>
          <p>
            Same pixel mapping, bailout and palette as the{" "}
            <a href="/wiki/mandelbrot">mandelbrot crate</a>. The image is a{" "}
            <code>[height, width]</code>{" "}
            tensor, and each tile program renders only its own tile. Shuffled
            launches show that finishing order means nothing; switch to row by
            row to compare.
          </p>
          <p>
            Turn on <em>Early-exit steps</em> to see how many iterations{" "}
            <code>mandelbrot_early_exit</code>{" "}
            would run per tile. A tile stops at the first check after all its
            pixels escape, so any tile touching the black set runs every step,
            just like <code>mandelbrot_fixed</code>.
          </p>
        </div>
        <TileMandelbrot />
      </section>

      <section class="demo" aria-labelledby="life-demo">
        <div class="demo-intro">
          <h2 id="life-demo">Life, with a ghost ring</h2>
          <p>
            The <code>life_step</code> kernel from the{" "}
            <a href="/wiki/life">life crate</a>, run tile program by tile
            program over a padded buffer. The dim border is the ghost ring. Its
            tile programs recompute the tiles on the opposite edge, so the world
            wraps without a copy pass.
          </p>
          <p>
            Point at any cell, or focus the grid and use the arrow keys. The
            orange block is the tile program that writes it, and the blue cells
            are the nine it sums, found the kernel's way: a view shift plus a
            block index. On a ghost tile, the dashed outline shows the interior
            tile it stands in for. Every generation is also checked against the
            crate's CPU reference. <a href="/wiki/stencils">Stencils</a>{" "}
            explains the trick.
          </p>
        </div>
        <LifeStencil />
      </section>
    </div>
  );
});
