export function renderGraphInfo(
  panel: Element,
  title: string,
  rows: ReadonlyArray<readonly [label: string, value: unknown]>,
) {
  const titleRow = document.createElement('div')
  const strong = document.createElement('strong')
  strong.textContent = title
  titleRow.appendChild(strong)

  const details = rows.map(([label, value]) => {
    const row = document.createElement('div')
    row.textContent = `${label}: ${String(value ?? '')}`
    return row
  })

  panel.replaceChildren(titleRow, ...details)
}
