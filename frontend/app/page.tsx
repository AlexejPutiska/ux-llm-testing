/**
 * Application root page.
 *
 * Immediately redirects to the About page (/o_projekte).
 * All meaningful content starts there.
 */
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/o_projekte");
}
