package projectstalker.build
import javax.inject.Inject
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations


abstract class CompileCudaTask extends DefaultTask {

    // --- PROPIEDADES DE LA TAREA ---

    @InputDirectory // Directorio de entrada con el código fuente
    abstract DirectoryProperty getSourceDir()

    @OutputDirectory // Directorio de salida para los archivos objeto (.o)
    abstract DirectoryProperty getOutputDir()

    // Inyección de dependencias: Gradle nos proporciona la herramienta para ejecutar comandos
    @Inject
    protected abstract ExecOperations getExecOperations()

    // --- ACCIÓN PRINCIPAL ---

    @TaskAction
    void compile() {
        // Aseguramos que el directorio de salida exista
        getOutputDir().get().asFile.mkdirs()

        // Buscamos todos los archivos .cu en el directorio de entrada
        project.fileTree(getSourceDir()).include('**/*.cu').each { sourceFile ->
            // Creamos un nombre único para el archivo objeto para evitar colisiones
            def objectFileName = sourceFile.name.replace('.cu', '.o')
            def objectFile = getOutputDir().file(objectFileName).get().asFile

            println "Compilando con CUDA: ${sourceFile.name} -> ${objectFile.name}"

            // Ejecutamos el compilador de NVIDIA (nvcc)
            getExecOperations().exec {
                executable 'nvcc'
                args(
                        '-c', // Compilar, no enlazar
                        '-I' + project.file('src/main/cpp/include').absolutePath,
                        '--compiler-options', '-fPIC',

                        // Genera código para tu GTX 1050 Ti (Pascal)
//                        '-gencode=arch=compute_61,code=sm_61',

                        // Genera código para tu objetivo de producción (Blackwell)
                        '-gencode=arch=compute_100,code=sm_100',

                        sourceFile.absolutePath, // Archivo de entrada
                        '-o', objectFile.absolutePath // Archivo de salida
                )
            }
        }
    }
}