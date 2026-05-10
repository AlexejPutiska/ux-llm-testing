/**
 * About page — static introductory page describing the project.
 *
 * Explains the purpose of the application and the types of UX tests it supports.
 * Serves as the landing page after the root redirect.
 */
export default function Page() {
  return (
    <div>
      <h1>O projekte</h1>
       <p>
        Táto webová aplikácia je súčasťou bakalárskej práce s názvom 
        „Vylepšovanie používateľského zážitku pomocou veľkých jazykových modelov“.
        Slúži na realizáciu UX testovania pomocou veľkých jazykových modelov.
      </p>
      <p>
        Aplikácia umožňuje vytvárať a spúšťať tri typy UX testov (preferenčný test, test prvého kliknutia, test spätnej väzby)
        zo snímok obrazovky. Výsledky generované jazykovým modelom sú ukladané a následne budú porovnávané s odpoveďami reálnych respondentov.
      </p>
    </div>
  );
}
