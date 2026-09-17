import { HttpError } from "fresh";
import { define } from "../../../utils.ts";
import { imageType, renderPath } from "../../../lib/tileworld.ts";

export const handler = define.handlers({
  async GET(ctx) {
    const path = renderPath(ctx.params.path);
    if (!path) throw new HttpError(404);
    try {
      const file = await Deno.open(path, { read: true });
      return new Response(file.readable, {
        headers: {
          "content-type": imageType(path)!,
          "cache-control": "no-cache",
        },
      });
    } catch (err) {
      if (err instanceof Deno.errors.NotFound) throw new HttpError(404);
      throw err;
    }
  },
});
