#include <GL/freeglut.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int offscreen_id;
int onscreen_id;

/*
 * The filename to which to save PNM screenshots.
 * Can be overridden by a commandline option.
 *
 * Save images with a ^S.
 */
const char *save_file_name = "offscreen.pnm";

static GLuint texName;

double theta = 0;


/*
 * Our text block.  80x25 is standard for a console, so why not?
 * Though with a 15-pixel font, you get more like 28x8.  The
 * terminal size should probably be computed at runtime from the
 * texture dimensions, which should be a user parameter.  Oh well.
 */
#define ROWS 25
#define COLS 80
char text[ ROWS ][ COLS+2 ];
#define OFFSCREEN_W 256
#define OFFSCREEN_H 128

void write_raw_pnm( const char *fname, char *pixels, int w, int h )
{
    FILE *f;

    f = fopen( fname, "wb" );
    if( !f )
        printf( "Ouch!  Cannot create file.\n" );
    else
    {
        int row;

        fprintf( f, "P6\n" );
        fprintf( f, "# CREATOR: offscreen freeglut demo\n" );
        fprintf( f, "%d %d\n", w, h );
        fprintf( f, "255\n" );

        /*
         * Write the rows in reverse order because OpenGL's 0th row
         * is at the bottom.
         */
        for( row = h; row; --row )
            fwrite( pixels + ((row - 1)*w*3), 1, 3 * w, f );

        fclose( f );
    }
}

void save_window( const char *file_name )
{
    char *pixels;
    int width  = glutGet( GLUT_WINDOW_WIDTH );
    int height = glutGet( GLUT_WINDOW_HEIGHT );
    pixels = malloc( 3 * width * height);
    if( pixels )
    {
        glPixelStorei( GL_PACK_ALIGNMENT, 1 );
        glReadPixels(
            0, 0, width, height,
            GL_RGB, GL_UNSIGNED_BYTE, (GLvoid *)pixels
        );
        write_raw_pnm( file_name, pixels, width, height );
    }
}


void cb_idle( void )
{
    glutSetWindow( onscreen_id );
    glutPostRedisplay( );
}

void cb_offscreen_display( void )
{
    static char pixels [OFFSCREEN_W * OFFSCREEN_H * 3];
    int i;

    glEnable( GL_DEPTH_TEST );

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glColor3d( 1.0, 0, 0 );
    glPushMatrix( );
    glRotated( .0071234 * glutGet( GLUT_ELAPSED_TIME ), 0, 1, 0 );
    glutSolidTorus( .2, .6, 20, 20 );
    glPopMatrix( );

    glDisable( GL_LIGHTING );
    glMatrixMode( GL_PROJECTION );
    glPushMatrix( );
    glLoadIdentity( );
    glColor3d( 0.0, 1.0, 0.0 );
    glRasterPos2f( -1, 1 - ( ( 2.0 * 10 ) / OFFSCREEN_H ) );
    glNormal3d( 0, 0, 1 );
    for( i = 7; i > -1; --i )
        glutBitmapString( GLUT_BITMAP_9_BY_15, text[ i ] );

    glPopMatrix( );
    glMatrixMode( GL_MODELVIEW );
    glEnable( GL_LIGHTING );

    glReadPixels(
        0, 0, OFFSCREEN_W, OFFSCREEN_H,
        GL_RGB, GL_UNSIGNED_BYTE, (GLvoid *)pixels
    );
    glutSetWindow( onscreen_id );
    glBindTexture( GL_TEXTURE_2D, texName );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexImage2D (
        GL_TEXTURE_2D, 0, GL_RGB, OFFSCREEN_W, OFFSCREEN_H,
        0, GL_RGB, GL_UNSIGNED_BYTE, (void *)pixels
    );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
    glDisable( GL_DEPTH_TEST );
}

void cb_onscreen_display( void )
{
    const static GLfloat flip_it_matrix[ 16 ] =
        { -1, 0, 0, 0,
          0, -1, 0, 0,
          0,  0, 1, 0,
          0,  0, 0, 1
        };

    glutSetWindow( offscreen_id );
    cb_offscreen_display( );

    glEnable( GL_DEPTH_TEST );
    glDisable( GL_LIGHTING );
    glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );
    glColor3d( 1.0, 0.0, 0.0 );
    glBegin( GL_LINE_LOOP );
    glVertex2d( 0.0, 0.0 );
    glVertex2d( 0.5, 0.0 );
    glVertex2d( 0.5, 0.5 );
    glVertex2d( 0.0, 0.5 );
    glVertex2d( 0.0, 0.0 );
    glEnd( );
    glColor3d( 1.0, 1.0, 1.0 );
    glRasterPos2d( -2.0, 0.8 );
    glutBitmapString(
        GLUT_BITMAP_TIMES_ROMAN_24, "Press ^S to Send to"
    );
    glRasterPos2d( -1.8, 0.6 );
    glutBitmapString( GLUT_BITMAP_TIMES_ROMAN_24, "'" );
    glutBitmapString(
        GLUT_BITMAP_9_BY_15, save_file_name );
    glutBitmapString( GLUT_BITMAP_TIMES_ROMAN_24, "'" );
    glRasterPos2d( -2.0, 0.3 );
    glutBitmapString(
        GLUT_BITMAP_TIMES_ROMAN_24, "Press Esc to quit."
    );

    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, texName );
    glBegin( GL_QUADS );
      glNormal3f( 0, 0, -1 );
      glTexCoord2i( 0, 0 ); glVertex3i( -9, -9, -4 );
      glTexCoord2i( 1, 0 ); glVertex3i(  9, -9, -4 );
      glTexCoord2i( 1, 1 ); glVertex3i(  9,  9, -4 );
      glTexCoord2i( 0, 1 ); glVertex3i( -9,  9, -4 );
      glTexCoord2i( 0, 0 ); glVertex3i( -9, -9, -4 );
    glEnd( );

    glEnable( GL_LIGHTING );
    glDisable( GL_LIGHTING );
    glPushMatrix( );
    theta = .01 * glutGet( GLUT_ELAPSED_TIME );
    glRotated( theta, 0, 1, 0 );
    glMatrixMode( GL_TEXTURE );
    glLoadMatrixf( flip_it_matrix );
    glMatrixMode( GL_MODELVIEW );
    glutSolidTeapot( 1.0 );
    glMatrixMode( GL_TEXTURE );
    glLoadIdentity( );
    glMatrixMode( GL_MODELVIEW );
    glDisable( GL_TEXTURE_2D );
    glPopMatrix( );
    glDisable( GL_DEPTH_TEST );

    glutSwapBuffers( );
}




static int cursor;
void vscroll( void )
{
    int i;
    for( i = ROWS-1; i; --i )
        strcpy( text[ i ], text[ i-1 ] );
    cursor = 0;
    text[ 0 ][ 0 ] = '\n';
    text[ 0 ][ 1 ] = 0;
}
void hscroll( void )
{
    int i = 0;
    if( text[ 0 ][ 0 ] )
        for( i = 0; i < COLS; ++i )
            text[ 0 ][ i ] = text[ 0 ][ i+1 ];
    text[ 0 ][ COLS ] = 0;
    --cursor;
    text[ 0 ][ cursor ] = '\n';
}
void add_char( char c )
{
    text[ 0 ][ cursor++ ] = c;
    text[ 0 ][ cursor   ] = '\n';
    text[ 0 ][ cursor+1 ] = 0;
    if( COLS <= cursor )
        hscroll( );
}



void cb_onscreen_keyboard( unsigned char key, int x, int y )
{
    int texture_updated = 1;

    switch( key )
    {
    case '\n':
    case '\r':
        vscroll( );
    case 'R' - '@': /* ^R */
        break;

    case 'S' - '@': /* ^S */
        save_window( save_file_name );
        texture_updated = 0;
        break;

    case '\x08': /* backspace */
        if( cursor )
        {
            --cursor;
            text[ 0 ][ cursor+1 ] = 0;
            text[ 0 ][ cursor   ] = '\n';
        }
        break;

    case '\x1b':
            exit( 0 );
        break;

    default:
        add_char( key );
        break;
    }
    glutSetWindow( offscreen_id );
    glutPostRedisplay( );
}


void cb_onscreen_reshape( int w, int h)
{
    double ar = w * 1.0/h;
    glEnable( GL_CULL_FACE );
    glCullFace( GL_BACK );
    glViewport( 0, 0, w, h );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity( );
    if( ar > 1 )
        glFrustum( -ar, ar, -1, 1, 2, 30 );
    else
        glFrustum( -1, 1, -1/ar, 1/ar, 2, 30 );
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity( );
    glTranslated( 0, 0, -5 );
}
void cb_offscreen_reshape( int w, int h)
{
    static GLfloat light_1_loc[ 3 ] = { 3, 4, 5 };
    static GLfloat light_1_col[ 3 ] = { 1, 1, 1 };
    static GLfloat material_diff[ 3 ] = { 1, 0, 0 };
    static GLfloat material_spec[ 3 ] = { 1, 1, 1 };
    double ar = w * 1.0/h;

    glEnable( GL_CULL_FACE );
    glViewport( 0, 0, w, h );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity( );
    if( ar > 1 )
        glFrustum( -ar, ar, -1, 1, 2, 30 );
    else
        glFrustum( -1, 1, -1/ar, 1/ar, 2, 30 );
    glTranslated( 0, 0, -3 );
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity( );
    glEnable( GL_LIGHT1 );
    glEnable( GL_LIGHTING );
    glLightfv( GL_LIGHT1, GL_POSITION, light_1_loc );
    glLightfv( GL_LIGHT1, GL_DIFFUSE, light_1_col );
    glLightfv( GL_LIGHT1, GL_SPECULAR, light_1_col );
    glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, material_diff );
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, material_spec );
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 90 );
}

int main( int argc, char **argv )
{
    glutInit( &argc, argv );
    if( argv[ 1 ] )
        save_file_name = argv[ 1 ];

    strcpy( text[ 4 ], "Touch type,\n" );
    strcpy( text[ 3 ], "on the\n" );
    strcpy( text[ 2 ], "teapot\n" );
    strcpy( text[ 1 ], "with me.\n" );
    strcpy( text[ 0 ], "\n" );
    glutInitDisplayMode( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );
    onscreen_id = glutCreateWindow( "Offscreen demo" );
    glutDisplayFunc( cb_onscreen_display );
    glutKeyboardFunc( cb_onscreen_keyboard );
    glutReshapeFunc( cb_onscreen_reshape );

    glutInitDisplayMode(
        GLUT_RGB | GLUT_SINGLE | GLUT_DEPTH | GLUT_OFFSCREEN
    );
    glutInitWindowSize( OFFSCREEN_W, OFFSCREEN_H );
    offscreen_id = glutCreateWindow( "" );
    glutDisplayFunc( cb_offscreen_display );
    cb_offscreen_reshape(  OFFSCREEN_W, OFFSCREEN_H );

    glGenTextures( 1, &texName );
    glBindTexture( GL_TEXTURE_2D, texName );

    glutIdleFunc( cb_idle );

    glutMainLoop( );
    return EXIT_SUCCESS;
}