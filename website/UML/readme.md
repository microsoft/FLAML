## UML Diagrams

We created all diagrams using PlantUML. If you are using Visual Studio Code,
you can leverage the PlantUML extension for modifying and previewing diagrams.

https://plantuml.com/

### Prerequisites

- PlantUML.jar: https://sourceforge.net/projects/plantuml/files/plantuml.jar/download
- Java: https://www.java.com/en/download/
    - Add to java executable to PATH or if using PlantUML extension for VS Code,
      add full path to `java.exe` under `plantuml.java` in PlantUML's `settings.json` file.
- Graphviz (not needed for sequence diagrams): https://graphviz.org/download/

Note: the latest versions of PlantUML include a minimalistic graphviz dot.exe.

### Building the diagrams

All diagram source codes are stored under `docs/UML/diagrams/src`, and outputs
under `docs/UML/diagrams/out`.

To automatically generate all diagrams run `make svg` for .svg outputs, or `make png` for .png outputs.

```bash
cd docs/UML
make svg
make png
```

To manually build the diagrams, you can use the following command, which will
search the directory for files with .pu extension with `@startuml` and `@enduml`, and
create all diagrams found under `docs/UML/diagrams/src`.

```bash
cd docs/UML
java -jar plantuml.jar -tsvg -o "../out" "diagrams/src"
```

Note: all diagram names are specified via `@startuml diagram_name` in each file.


### Contribution

To modify an existing diagram simply modify between `@startuml diagram_name` and `@enduml`,
and regenerate diagrams. To create new diagrams, please create a new .pu file and it will
automatically be detected when calling `make svg`.