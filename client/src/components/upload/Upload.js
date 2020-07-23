import React, { Fragment } from 'react';
import {
    Button, TextField, Grid
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles';
import { textAlign } from '@material-ui/system';

const useStyles = makeStyles(theme => ({
    container: {
        minWidth: "100vh"
    },

    text: {
        width: '100%',
    },

    submit: {
        height: '35px',
        fontFamily: 'Montserrat',
    },

}));

const Upload = () => {
    const classes = useStyles();

    const getAnnotatedDocument = () => {
        fetch("http://localhost:5000/classify")
            .then(res => res.json())
            .then(data => console.log(data))

    }

    return (
        <div>
            <Grid
                container
                direction="column"
                alignItems="center"
                justify="center"
                className={classes.container}
            >
                <Grid item xs={12}>
                    <TextField id="outlined-basic" label="Enter local path..." variant="outlined" className={classes.text} margin='dense' autoFocus />
                </Grid>

                <Grid item xs={12}>
                    <Button type="submit" variant="contained" color="primary" className={classes.submit} onClick={getAnnotatedDocument}>Submit</Button>
                </Grid>
            </Grid>
        </div>
    )
};

export default Upload;