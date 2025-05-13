import html
import os


class HTMLTable:
    def __init__(self, param_names, output_file="RESULTS.html"):
        self.param_names = param_names
        self.output_file = output_file
        self.rows = []

    def add_a_row(self, **kwargs):
        row = []
        for param in self.param_names:
            value = kwargs.get(param, "")
            if param == "file_name":
                full_path = os.path.abspath(value)
                print(full_path)
                fname = full_path.replace("\\", "/").replace(" ", "%20")
                hf = '<a href="file://' + fname + '" >' + value + "</a><br> "
                row.append(hf)
            else:
                row.append(html.escape(str(value)))
        self.rows.append(row)

    def save(self):
        html_content = self._generate_html()
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_html(self):
        # JavaScript for alphabetical sorting
        js_sort = """
        <script>
        function sortTable(n) {
            const table = document.getElementById("dataTable");
            let switching = true;
            let dir = "asc";
            while (switching) {
                switching = false;
                let rows = table.rows;
                let shouldSwitch = false;
                let i=1;
                for ( i = 1; i < rows.length - 1; i++) {
                    shouldSwitch = false;
                    let x = rows[i].getElementsByTagName("TD")[n].innerText.toLowerCase();
                    let y = rows[i + 1].getElementsByTagName("TD")[n].innerText.toLowerCase();
                    if ((dir === "asc" && x > y) || (dir === "desc" && x < y)) {
                        shouldSwitch = true;
                        break;
                    }
                }
                if (shouldSwitch) {
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                } else {
                    if (dir === "asc") {
                        dir = "desc";
                    } else {
                        dir = "asc";
                    }
                }
            }
        }
        </script>
        """

        # HTML for table headers
        header_html = "".join(
            f'<th onclick="sortTable({i})" style="cursor:pointer;">{html.escape(name)}</th>'
            for i, name in enumerate(self.param_names)
        )

        # HTML for table rows
        rows_html = ""
        for row in self.rows:
            rows_html += (
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sortable Table</title>
    <style>
        table, th, td {{
            border: 1px solid black;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
    {js_sort}
</head>
<body>
    <table id="dataTable">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</body>
</html>
"""


# Example usage:
if __name__ == "__main__":
    table = HTMLTable(["name", "age", "city"])
    table.add_a_row(name="Alice", age="30", city="New York")
    table.add_a_row(name="Bob", age="25", city="Los Angeles")
    table.add_a_row(name="Charlie", age="35", city="Chicago")
    table.save()
